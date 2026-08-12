import json
import sqlite3
import struct
import warnings
import uuid
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from Tools import LogSetting

logger = LogSetting.create(__name__)


def _serialize_f32(vector: List[float]) -> bytes:
    """将浮点数列表序列化为紧凑的“原始字节”格式

    来源:
        https://github.com/asg017/sqlite-vec/blob/21c5a14fc71c83f135f5b00c84115139fd12c492/examples/simple-python/demo.py#L8-L10
    """
    return struct.pack("%sf" % len(vector), *vector)


class CustomizeSQLiteVec(VectorStore):
    """
    自定义的SQLiteVec向量数据库
    """

    def __init__(
            self,
            table: str,
            connection: Optional[sqlite3.Connection],
            embedding: Embeddings,
            db_file: str = "vec.db",
            is_async: Optional[bool] = False,
    ):
        """使用扩展名为vss的sqlite客户端进行初始化。"""
        try:
            import sqlite_vec  # noqa  # pylint: disable=unused-import
        except ImportError:
            raise ImportError(
                "Could not import sqlite-vec python package. "
                "Please install it with `pip install sqlite-vec`."
            )

        if not connection:
            connection = self.create_connection(db_file, is_async)

        if not isinstance(embedding, Embeddings):
            warnings.warn("embeddings input must be Embeddings object.")

        self._connection = connection
        self._table = table
        self._embedding = embedding

        self.create_table_if_not_exists()

    def create_table_if_not_exists(self) -> None:
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table}
            (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE,
                text TEXT,
                metadata BLOB,
                text_embedding BLOB
            )
            ;
            """
        )
        self._connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self._table}_vec USING vec0(
                rowid INTEGER PRIMARY KEY,
                text_embedding float[{self.get_dimensionality()}]
                distance_metric=cosine
            )
            ;
            """
        )
        self._connection.execute(
            f"""
                CREATE TRIGGER IF NOT EXISTS {self._table}_embed_text 
                AFTER INSERT ON {self._table}
                BEGIN
                    INSERT INTO {self._table}_vec(rowid, text_embedding)
                    VALUES (new.rowid, new.text_embedding) 
                    ;
                END;
            """
        )
        self._connection.execute(
            f"""
                CREATE TRIGGER IF NOT EXISTS {self._table}_delete_vec
                AFTER DELETE ON {self._table}
                BEGIN
                    DELETE FROM {self._table}_vec WHERE rowid = old.rowid;
                END;
                """
        )
        self._connection.commit()

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """向向量库索引添加文本。
        参数:
            texts: 可将字符串添加到向量库中。
            metadatas: 与文本关联的元数据的可选列表。
            ids: 自定义ID列表，用于标记片段的的唯一标识
            kwargs: 向量库其他参数
        """
        max_id = self._connection.execute(
            f"SELECT max(rowid) as rowid FROM {self._table}"
        ).fetchone()["rowid"]
        if max_id is None:  # no text added yet
            max_id = 0

        embeds = self._embedding.embed_documents(list(texts))
        if not metadatas:
            metadatas = [{} for _ in texts]
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        data_input = [
            (id, text, json.dumps(metadata), _serialize_f32(embed))
            for id, text, metadata, embed in zip(ids, texts, metadatas, embeds)
        ]
        self._connection.executemany(
            f"INSERT INTO {self._table}(id, text, metadata, text_embedding) VALUES (?,?,?,?)",
            data_input,
        )
        self._connection.commit()
        # 提取我们刚刚插入的每个id
        results = self._connection.execute(
            f"SELECT rowid FROM {self._table} WHERE rowid > {max_id}"
        )
        return [str(row["rowid"]) for row in results]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if not ids:
            raise ValueError("Must provide ids to delete.")
        placeholders = ",".join(["?" for _ in ids])
        self._connection.execute(
            f"DELETE FROM {self._table} WHERE id IN ({placeholders})", ids
        )
        self._connection.commit()
        return True

    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """返回与嵌入最相似的文档，可选择按元数据过滤。

        参数:
            embedding: 要搜索的嵌入向量。
            k: 要返回的结果数。
            filter: {metadata_key:value}条件的可选字典，匹配
                通过JSON_lixt对JSON元数据进行处理，例如:
                ``{“user_id”：“u1”}`。
        """
        filter_clause = ""
        params: list = [_serialize_f32(embedding), k]
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(f"json_extract(e.metadata, '$.{key}') = ?")
                params.append(value)
            if conditions:
                filter_clause = " AND " + " AND ".join(conditions)

        sql_query = f"""
            SELECT
                id,
                text,
                metadata,
                distance
            FROM {self._table} AS e
            INNER JOIN {self._table}_vec AS v on v.rowid = e.rowid
            WHERE
                v.text_embedding MATCH ?
                AND k = ?
                {filter_clause}
            ORDER BY distance
        """
        cursor = self._connection.cursor()
        cursor.execute(
            sql_query,
            params,
        )
        results = cursor.fetchall()

        documents = []
        for row in results:
            metadata = json.loads(row["metadata"]) or {}
            doc = Document(page_content=row["text"], metadata=metadata, id=row["id"])
            documents.append((doc, row["distance"]))

        return documents

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k
        )
        return [doc for doc, _ in documents]

    def similarity_search_with_score(
            self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """返回与查询最相似的文档。"""
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return documents

    def similarity_search_by_vector(
            self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k
        )
        return [doc for doc, _ in documents]

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            table: str = "langchain",
            db_file: str = "vec.db",
            **kwargs: Any,
    ) -> "CustomizeSQLiteVec":
        """返回从文本和嵌入初始化的CustomizeSQLiteVec。"""
        connection = cls.create_connection(db_file)
        vec = cls(
            table=table, connection=connection, db_file=db_file, embedding=embedding
        )
        vec.add_texts(texts=texts, metadatas=metadatas)
        return vec

    @staticmethod
    def create_connection(db_file: str, is_async: bool = False) -> sqlite3.Connection:
        import sqlite3
        import sqlite_vec

        check = not is_async
        connection = sqlite3.connect(db_file, check_same_thread=check)
        connection.row_factory = sqlite3.Row
        # 并发优化：WAL 允许读写并发，busy_timeout 撞锁时等待而不是立即报 database is locked
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        return connection

    def get_dimensionality(self) -> int:
        """
        执行虚拟嵌入以计算维度数量的函数此嵌入函数返回。需要虚拟表DDL。
        """
        dummy_text = "This is a dummy text"
        dummy_embedding = self._embedding.embed_query(dummy_text)
        return len(dummy_embedding)

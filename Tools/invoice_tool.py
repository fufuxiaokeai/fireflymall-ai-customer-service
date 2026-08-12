import datetime
import os
from typing import TypedDict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
)

from Tools.after_sales_tool import get_id
from Tools.log_settings import LogSetting

logger = LogSetting.create(__name__)


class ItemsDict(TypedDict):
    name: str
    price: float
    quantity: int
    total_amount: float


def _register_font():
    """注册中文字体，解决中文乱码"""
    font_path = "C:/Windows/Fonts/simsun.ttc"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/simhei.ttf"

    try:
        pdfmetrics.registerFont(TTFont('SimSun', font_path))
        return 'SimSun'
    except Exception as e:
        logger.warning(f"警告：未找到中文字体，中文可能显示为方框，错误信息：{e}")
        return 'Helvetica'


class InvoiceTool:
    def __init__(self, items: List[ItemsDict]):
        if not items:
            raise ValueError('items cannot be empty')
        self.invoice_data = {
            "invoice_no": f"FP{get_id()}",
            "date": datetime.date.today().strftime("%Y年%m月%d日"),
            "drawer": "张三",

            "buyer": {
                "name": "个人",
                "tax_id": "",
                "address": "",
                "phone": "415411"
            },

            "seller": {
                "name": "某某科技有限公司",
                "tax_id": "91440000XXXXXXXXXX",
                "address": "",
                "phone": "114514"
            },

            "items": items,

            "remark": "本发票仅作演示使用，不作报销凭证。"
        }

    def generate_invoice(self, output_path):
        output_path = str(output_path)
        font_name = _register_font()
        line_color = colors.darkred

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm
        )

        styles = getSampleStyleSheet()
        story = []

        # ========== 通用样式 ==========
        title_style = ParagraphStyle(
            'title',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=22,
            alignment=1,
            textColor=line_color
        )

        normal_style = ParagraphStyle(
            'normal',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=10,
            leading=16
        )

        right_style = ParagraphStyle(
            'right',
            parent=normal_style,
            alignment=2
        )

        label_style = ParagraphStyle(
            'label',
            parent=normal_style,
            textColor=line_color
        )

        # 居中单元格样式（序号、单价、数量、金额列）
        cell_center_style = ParagraphStyle(
            'cell_center',
            parent=normal_style,
            alignment=1,
            leading=14
        )
        # 左对齐单元格样式（商品名称列）
        cell_left_style = ParagraphStyle(
            'cell_left',
            parent=normal_style,
            alignment=0,
            leading=14
        )
        # 表头样式
        cell_header_style = ParagraphStyle(
            'cell_header',
            parent=cell_center_style,
            textColor=line_color
        )

        # ========== 顶部标题 + 发票号日期 ==========
        header_table = Table(
            [
                [
                    "",
                    Paragraph("电子发票（普通发票）", title_style),
                    Paragraph(
                        f"发票编号：{self.invoice_data['invoice_no']}<br/>开票日期：{self.invoice_data['date']}",
                        right_style
                    )
                ]
            ],
            colWidths=[25 * mm, 90 * mm, 55 * mm]
        )

        header_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEBELOW', (1, 0), (1, 0), 1, line_color),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))

        story.append(header_table)
        story.append(Spacer(1, 8 * mm))

        # ========== 购销双方信息（修复等高对齐） ==========
        # 购买方内容
        buyer_lines = [
            f"名称：{self.invoice_data['buyer']['name']}",
            f"统一社会信用代码：{self.invoice_data['buyer'].get('tax_id', '无')}"
        ]
        if self.invoice_data['buyer'].get('address'):
            buyer_lines.append(f"地址：{self.invoice_data['buyer']['address']}")
        if self.invoice_data['buyer'].get('phone'):
            buyer_lines.append(f"电话：{self.invoice_data['buyer']['phone']}")
        buyer_content = Paragraph("<br/>".join(buyer_lines), normal_style)

        # 销售方内容
        seller_lines = [
            f"名称：{self.invoice_data['seller']['name']}",
            f"统一社会信用代码：{self.invoice_data['seller'].get('tax_id', '无')}"
        ]
        if self.invoice_data['seller'].get('address'):
            seller_lines.append(f"地址：{self.invoice_data['seller']['address']}")
        if self.invoice_data['seller'].get('phone'):
            seller_lines.append(f"电话：{self.invoice_data['seller']['phone']}")
        seller_content = Paragraph("<br/>".join(seller_lines), normal_style)

        # 左右两栏放在同一行表格内，自动等高
        party_data = [
            [
                Paragraph("购买方信息", label_style),
                buyer_content,
                Paragraph("销售方信息", label_style),
                seller_content
            ]
        ]
        party_table = Table(
            party_data,
            colWidths=[22 * mm, 63 * mm, 22 * mm, 63 * mm]
        )
        party_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            # 外边框
            ('BOX', (0, 0), (-1, -1), 0.8, line_color),
            # 中间竖线（左右栏分隔）
            ('LINEAFTER', (1, 0), (1, 0), 0.8, line_color),
            # 标签栏右侧竖线
            ('LINEAFTER', (0, 0), (0, 0), 0.5, line_color),
            ('LINEAFTER', (2, 0), (2, 0), 0.5, line_color),
        ]))

        story.append(party_table)
        story.append(Spacer(1, 8 * mm))

        # ========== 商品明细表格 ==========
        table_header = [
            Paragraph("序号", cell_header_style),
            Paragraph("商品名称", cell_header_style),
            Paragraph("单价(元)", cell_header_style),
            Paragraph("数量", cell_header_style),
            Paragraph("金额(元)", cell_header_style)
        ]
        table_data = [table_header]

        total_amount = 0
        for idx, item in enumerate(self.invoice_data['items'], 1):
            amount = item.get('total_amount', 0)
            total_amount += amount
            table_data.append([
                Paragraph(str(idx), cell_center_style),
                Paragraph(item['name'], cell_left_style),
                Paragraph(f"{item['price']:.2f}", cell_center_style),
                Paragraph(str(item['quantity']), cell_center_style),
                Paragraph(f"{amount:.2f}", cell_center_style)
            ])

        table_data.append([
            Paragraph("", cell_center_style),
            Paragraph("", cell_center_style),
            Paragraph("", cell_center_style),
            Paragraph("合计", right_style),
            Paragraph(f"{total_amount:.2f}", right_style)
        ])

        col_widths = [15 * mm, 70 * mm, 25 * mm, 20 * mm, 40 * mm]

        item_table = Table(table_data, colWidths=col_widths)
        item_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -2), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, line_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), line_color),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('ALIGN', (-2, -1), (-1, -1), 'RIGHT'),
        ]))

        story.append(item_table)
        story.append(Spacer(1, 2 * mm))

        # ========== 价税合计 ==========
        def num_to_cn(num):
            cn = "零壹贰叁肆伍陆柒捌玖"
            unit = ["", "拾", "佰", "仟", "万", "拾", "佰", "仟", "亿"]
            integer = int(num)
            decimal = round((num - integer) * 100)

            int_str = ""
            if integer == 0:
                int_str = "零"
            else:
                s = str(integer)
                for i, n in enumerate(s):
                    pos = len(s) - i - 1
                    if n != '0':
                        int_str += cn[int(n)] + unit[pos]
                    else:
                        if int_str and int_str[-1] != '零':
                            int_str += '零'
                int_str = int_str.rstrip('零')

            dec_str = ""
            if decimal > 0:
                jiao = decimal // 10
                fen = decimal % 10
                if jiao > 0:
                    dec_str += cn[jiao] + "角"
                if fen > 0:
                    dec_str += cn[fen] + "分"
            else:
                dec_str = "整"

            return f"{int_str}元{dec_str}"

        total_cn = num_to_cn(total_amount)

        total_table = Table(
            [
                [
                    Paragraph(f"价税合计（大写）：{total_cn}", normal_style),
                    Paragraph(f"（小写）¥{total_amount:.2f}", right_style)
                ]
            ],
            colWidths=[120 * mm, 50 * mm]
        )

        total_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('BOX', (0, 0), (-1, -1), 0.5, line_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))

        story.append(total_table)
        story.append(Spacer(1, 2 * mm))

        # ========== 备注栏 ==========
        remark_table = Table(
            [
                [
                    Paragraph("备注", label_style),
                    Paragraph(self.invoice_data.get('remark', '无'), normal_style)
                ]
            ],
            colWidths=[22 * mm, 148 * mm],
            rowHeights=[30 * mm]
        )

        remark_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 0.5, line_color),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('LINEAFTER', (0, 0), (0, 0), 0.5, line_color),
        ]))

        story.append(remark_table)
        story.append(Spacer(1, 15 * mm))

        # ========== 底部开票人 ==========
        story.append(Paragraph(f"开票人：{self.invoice_data.get('drawer', '管理员')}", normal_style))

        doc.build(story)
        print(f"发票已生成：{output_path}")


if __name__ == '__main__':
    invoice = InvoiceTool(items=[{
        'name': '商品1',
        'price': 100,
        'quantity': 2,
        'total_amount': 200
    }])
    invoice.generate_invoice('test.pdf')
    pass
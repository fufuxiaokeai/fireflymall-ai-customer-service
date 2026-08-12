import os
import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr

from langchain_core.tools import tool

from Tools import LogSetting
from Tools.registry import register_tool
from load_config.config import config

logger = LogSetting.create(__name__)

_email_config = config.get('email')
if not _email_config:
    raise ValueError('邮箱配置不能为空')
_stmp = _email_config.get('smtp')
_port = int(_email_config.get('port'))
_sender = _email_config.get('sender')
_sender_name = _email_config.get('sender_name')
_receiver = _email_config.get('receiver')

if isinstance(_receiver, list):
    if len(_receiver) == 1:
        _receiver = _receiver[0]
    else:
        _receiver = ','.join(_receiver)


def send_error_email(
        subject: str,
        content: str,
):
    """
    发送错误消息的邮件

    :param subject: 邮件主题
    :param content: 邮件内容
    """
    msg = MIMEText(_text=content, _subtype="plain")
    msg['From'] = formataddr((_sender_name, _sender))
    msg['To'] = Header(_receiver, charset='utf-8')
    msg['Subject'] = Header(f"AI 客服服务引发了错误信息：{subject}", charset='utf-8')

    try:
        with smtplib.SMTP(_stmp, _port) as server:
            server.login(_sender, os.getenv("EMAIL_AUTH_CODE"))
            server.sendmail(_sender, _receiver, msg.as_string())
            logger.info("邮件发送成功")
        return True
    except smtplib.SMTPException as e:
        logger.error(f"邮件发送失败，错误：{e}")
        return False


@register_tool('main_agent')
@tool
def send_error_email_tool(
        subject: str,
        content: str,
):
    """
    该工具用于将当前的发生错误消息发送到技术人员的邮箱中，若在运行当中引发的错误，可以通过此工具发送错误信息
    参数：
        subject: 主题（标题） - 用于概要当前错误的简单标题
        content: 错误的详细消息
    """
    return "邮件发送成功" if send_error_email(subject, content) else "邮件发送失败"

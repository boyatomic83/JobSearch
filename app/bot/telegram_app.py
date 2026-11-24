"""python-telegram-bot adapter: wires Telegram updates to app.bot.service and sends `Out`s.

Only translation lives here (update → service call → messages); the logic is in
service.py so it can be tested without Telegram.
"""

import hashlib
import io
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from app.bot import service
from app.bot.format import split_message
from app.bot.service import Deps, Out
from app.config import get_settings

logger = logging.getLogger(__name__)

ROLE, LOCATION, REMOTE, SALARY = range(4)
MAX_PDF_BYTES = 5 * 1024 * 1024


def webhook_secret(token: str) -> str:
    """Telegram echoes this in X-Telegram-Bot-Api-Secret-Token on every webhook call.
    Derived from the bot token so there's no extra secret to configure."""
    return hashlib.sha256(f"jobscout-webhook:{token}".encode()).hexdigest()


def _deps(context: ContextTypes.DEFAULT_TYPE) -> Deps:
    return context.application.bot_data["deps"]


async def send(bot, chat_id: int, outs: list[Out]) -> None:
    for out in outs:
        if out.document:
            name, data = out.document
            await bot.send_document(chat_id, InputFile(io.BytesIO(data), filename=name))
            continue
        if not out.text:
            continue
        chunks = split_message(out.text)
        markup = (
            InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton(label, callback_data=data) for label, data in row]
                    for row in out.buttons
                ]  # fmt: skip
            )
            if out.buttons
            else None
        )
        for i, chunk in enumerate(chunks):
            await bot.send_message(
                chat_id, chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True,
                reply_markup=markup if i == len(chunks) - 1 else None,
            )  # fmt: skip


async def _user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat = update.effective_chat
    name = update.effective_user.full_name if update.effective_user else None
    return await service.get_or_create_user(_deps(context), chat.id, name)


async def _typing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)


# ---- onboarding (/start) ------------------------------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await _user_id(update, context)
    await update.message.reply_html(
        "👋 Hi! I'm <b>JobScout</b>. I'll send you the best real jobs for you every morning.\n\n"
        "4 quick questions (send /cancel to skip).\n\n"
        "<b>1/4</b> What role(s) are you looking for? e.g. <i>Backend Engineer, ML Engineer</i>"
    )
    return ROLE


async def on_role(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await service.save_onboarding(_deps(context), await _user_id(update, context), "role",
                                  update.message.text)  # fmt: skip
    await update.message.reply_html(
        "<b>2/4</b> Where? City and country, e.g. <i>Bangalore, India</i>"
    )
    return LOCATION


async def on_location(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await service.save_onboarding(_deps(context), await _user_id(update, context), "location",
                                  update.message.text)  # fmt: skip
    buttons = InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 Remote only", callback_data="onb:remote_only")],
        [InlineKeyboardButton("🏢 On-site / hybrid is fine too", callback_data="onb:any")],
    ])  # fmt: skip
    await update.message.reply_html("<b>3/4</b> Remote?", reply_markup=buttons)
    return REMOTE


async def on_remote(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    choice = query.data.split(":", 1)[1]
    await service.save_onboarding(_deps(context), await _user_id(update, context), "remote", choice)
    await query.edit_message_text("Remote only ✅" if choice == "remote_only" else "Any setup ✅")
    await context.bot.send_message(
        update.effective_chat.id,
        "<b>4/4</b> Minimum yearly salary? e.g. <i>40 LPA</i>, <i>$150k</i>, or <i>skip</i>",
        parse_mode=ParseMode.HTML,
    )
    return SALARY


async def on_salary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    deps = _deps(context)
    user_id = await _user_id(update, context)
    error = await service.save_onboarding(deps, user_id, "salary", update.message.text)
    if error:
        await update.message.reply_html(error)
        return SALARY
    await update.message.reply_html(
        await service.prefs_text(deps, user_id)
        + "\n\n📄 Now send me your resume as a <b>PDF</b> and I'll find your matches."
    )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_html("OK. Send /help any time.")
    return ConversationHandler.END


# ---- commands --------------------------------------------------------------------------


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(service.HELP, disable_web_page_preview=True)


async def resume_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html("📄 Send your resume as a <b>PDF</b> file (max 5 MB).")


async def prefs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = await service.prefs_text(_deps(context), await _user_id(update, context))
    await update.message.reply_html(text)


async def jobs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _typing(update, context)
    outs = await service.jobs(_deps(context), await _user_id(update, context))
    await send(context.bot, update.effective_chat.id, outs)


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if doc.mime_type != "application/pdf":
        await update.message.reply_html("Please send your resume as a <b>PDF</b>.")
        return
    if doc.file_size and doc.file_size > MAX_PDF_BYTES:
        await update.message.reply_html("That PDF is over 5 MB — please send a smaller one.")
        return
    await update.message.reply_html("📄 Reading your resume…")
    await _typing(update, context)
    data = bytes(await (await doc.get_file()).download_as_bytearray())
    outs = await service.handle_resume(_deps(context), await _user_id(update, context),
                                       doc.file_name or "resume.pdf", data)  # fmt: skip
    await send(context.bot, update.effective_chat.id, outs)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    deps = _deps(context)
    user_id = await _user_id(update, context)
    await _typing(update, context)
    if (match_id := context.user_data.pop("awaiting_reason", None)) is not None:
        outs = await service.feedback_text(deps, user_id, match_id, update.message.text)
    else:
        outs = await service.chat(deps, user_id, update.message.text)
    await send(context.bot, update.effective_chat.id, outs)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    deps = _deps(context)
    user_id = await _user_id(update, context)
    kind, *rest = query.data.split(":")
    if kind == "fb":
        outs = await service.feedback(deps, user_id, int(rest[0]), int(rest[1]))
    elif kind == "why":
        match_id, code = int(rest[0]), rest[1]
        if code == "other":
            context.user_data["awaiting_reason"] = match_id
        outs = await service.feedback_reason(deps, user_id, match_id, code)
    elif kind == "tailor":
        await context.bot.send_message(update.effective_chat.id, "📝 Tailoring your resume…")
        await _typing(update, context)
        outs = await service.tailor(deps, user_id, int(rest[0]))
    else:
        outs = []
    await send(context.bot, update.effective_chat.id, outs)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("telegram handler failed", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        await context.bot.send_message(
            update.effective_chat.id, "Sorry, something went wrong. Please try again."
        )


def build_application(deps: Deps) -> Application:
    token = get_settings().telegram_bot_token
    # concurrent_updates: one user's slow LLM call must not block everyone else.
    app = ApplicationBuilder().token(token).concurrent_updates(8).build()
    app.bot_data["deps"] = deps
    text = filters.TEXT & ~filters.COMMAND
    app.add_handler(
        ConversationHandler(
            entry_points=[CommandHandler("start", start)],
            states={
                ROLE: [MessageHandler(text, on_role)],
                LOCATION: [MessageHandler(text, on_location)],
                REMOTE: [CallbackQueryHandler(on_remote, pattern=r"^onb:")],
                SALARY: [MessageHandler(text, on_salary)],
            },
            fallbacks=[CommandHandler("cancel", cancel)],
        )
    )
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("resume", resume_cmd))
    app.add_handler(CommandHandler("prefs", prefs_cmd))
    app.add_handler(CommandHandler("jobs", jobs_cmd))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^(fb|why|tailor):"))
    app.add_handler(MessageHandler(text, on_text))
    app.add_error_handler(on_error)
    return app

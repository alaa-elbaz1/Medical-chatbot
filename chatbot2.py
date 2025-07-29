import streamlit as st
import pandas as pd
import hashlib
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# إعداد مفاتيح Gemini
GEMINI_API_KEY = "AIzaSyBBWg_3GoVzBRQuarjkIoIPkCNeW6xJgEY"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-pro")

# إعداد GPT-2
@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

gpt2_tokenizer, gpt2_model = load_gpt2()

# إعداد المجلدات
DATA_DIR = Path("app_data")
DATA_DIR.mkdir(exist_ok=True)
USERS_PATH = DATA_DIR / "users.csv"
HISTORY_DIR = DATA_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)

# اختيار اللغة
lang = st.sidebar.selectbox("اختر اللغة | Choose Language", ["العربية", "English"])
AR = lang == "العربية"
def _(ar, en): return ar if AR else en

# تحميل بيانات الأسئلة
DATA_FILE = "train.csv"
try:
    data = pd.read_csv(DATA_FILE, encoding="utf-8")
    if 'Question' not in data.columns or 'Answer' not in data.columns:
        st.error(_("يجب أن يحتوي الملف على أعمدة: Question و Answer.", "Dataset must contain 'Question' and 'Answer' columns."))
        data = pd.DataFrame(columns=["Question", "Answer"])
    else:
        data.dropna(subset=["Question", "Answer"], inplace=True)
except FileNotFoundError:
    st.error(_(f"الملف '{DATA_FILE}' غير موجود.", f"File '{DATA_FILE}' not found."))
    data = pd.DataFrame(columns=["Question", "Answer"])

# تشفير كلمة المرور
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    try:
        if USERS_PATH.exists() and os.path.getsize(USERS_PATH) > 0:
            return pd.read_csv(USERS_PATH)
        return pd.DataFrame(columns=["username", "password"])
    except Exception as e:
        st.error(_(f"خطأ في تحميل بيانات المستخدمين: {e}", f"Error loading users data: {e}"))
        return pd.DataFrame(columns=["username", "password"])

def write_users(users_df):
    try:
        users_df.to_csv(USERS_PATH, index=False, encoding='utf-8')
        return True
    except Exception as e:
        st.error(f"{_('خطأ أثناء حفظ المستخدمين', 'Error saving users')}: {e}")
        return False

def save_user(username, password):
    try:
        users = load_users()
        if username.strip() == "" or password.strip() == "":
            st.error(_("لا يمكن ترك اسم المستخدم أو كلمة المرور فارغة.", "Username and password cannot be empty."))
            return False
        if username in users['username'].values:
            st.error(_("اسم المستخدم موجود مسبقاً.", "Username already exists."))
            return False
        hashed_password = hash_password(password)
        new_user = pd.DataFrame([[username, hashed_password]], columns=["username", "password"])
        users = pd.concat([users, new_user], ignore_index=True)
        return write_users(users)
    except Exception as e:
        st.error(_(f"خطأ أثناء حفظ المستخدم: {e}", f"Error saving user: {e}"))
        return False

def save_to_history(username, question, answer):
    try:
        history_file = HISTORY_DIR / f"{username}.csv"
        row = pd.DataFrame([[question, answer]], columns=["Question", "Answer"])
        if history_file.exists():
            existing = pd.read_csv(history_file)
            full = pd.concat([existing, row], ignore_index=True)
        else:
            full = row
        full.to_csv(history_file, index=False, encoding="utf-8")
    except Exception as e:
        st.error(_(f"خطأ في حفظ السجل: {e}", f"Error saving history: {e}"))

def show_history(username, translate):
    try:
        history_file = HISTORY_DIR / f"{username}.csv"
        if history_file.exists():
            st.subheader(translate("سجل الأسئلة", "Your Question History"))
            history = pd.read_csv(history_file)
            for _, row in history.iloc[::-1].iterrows():
                st.markdown(f"**{translate('أنت', 'You')}:** {row['Question']}")
                st.markdown(f"**{translate('البوت', 'Bot')}:** {row['Answer']}")
                st.markdown("---")
    except Exception as e:
        st.error(translate(f"خطأ في عرض السجل: {e}", f"Error displaying history: {e}"))

def ask_gpt2(prompt, max_length=100):
    try:
        inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
        outputs = gpt2_model.generate(inputs, max_length=max_length, pad_token_id=gpt2_tokenizer.eos_token_id)
        answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        st.error(f"GPT-2 Error: {e}")
        return None

def ask_gemini(message):
    try:
        response = gemini_model.generate_content(message)
        return response.text
    except Exception as e:
        st.error(_(f"خطأ في الاتصال ب Gemini: {e}", f"Gemini connection error: {e}"))
        return _("عذراً، حدث خطأ أثناء معالجة سؤالك", "Sorry, an error occurred while processing your question")

def ask_combined_model(message):
    gpt2_response = ask_gpt2(message)
    if not gpt2_response or len(gpt2_response.split()) < 10:
        st.info(_("جارٍ استخدام Gemini لتحسين الإجابة...", "Using Gemini for better response..."))
        return ask_gemini(message)
    return gpt2_response

def chatbot_page():
    st.title(f"{'مرحباً' if AR else 'Welcome'}, {st.session_state.username}")
    question = st.text_input(_("اكتب سؤالك الطبي هنا", "Ask your medical question here"))
    if question and st.button(_("إرسال", "Send"), type="primary"):
        if data.empty:
            st.warning(_("لا تتوفر بيانات للإجابة", "No data available to answer."))
            return
        try:
            user_input = question.strip().lower()
            questions = data['Question'].fillna("").str.lower().str.strip()
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(questions)
            user_vec = vectorizer.transform([user_input])
            similarities = cosine_similarity(user_vec, X)
            index = similarities.argmax()
            score = similarities[0][index]
            if score > 0.4:
                answer = data['Answer'].iloc[index]
            else:
                answer = ask_combined_model(question)
            st.success(answer)
            save_to_history(st.session_state.username, question, answer)
        except Exception as e:
            st.error(f"{_('حدث خطأ أثناء المعالجة', 'An error occurred during processing')}: {e}")
    show_history(st.session_state.username, _)

def login_page():
    st.title(_("تسجيل الدخول", "Login"))
    with st.form("login_form"):
        username = st.text_input(_("اسم المستخدم", "Username"))
        password = st.text_input(_("كلمة المرور", "Password"), type="password")
        if st.form_submit_button(_("تسجيل الدخول", "Login")):
            if not username.strip() or not password.strip():
                st.error(_("الرجاء إدخال جميع البيانات", "Please enter all fields."))
                return
            try:
                users = load_users()
                hashed = hash_password(password)
                match = users[(users['username'] == username) & (users['password'] == hashed)]
                if not match.empty:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error(_("اسم المستخدم أو كلمة المرور غير صحيحة", "Incorrect username or password."))
            except Exception as e:
                st.error(_(f"خطأ في تسجيل الدخول: {e}", f"Login error: {e}"))

def signup_page():
    st.title(_("إنشاء حساب جديد", "Sign Up"))
    with st.form("signup_form"):
        username = st.text_input(_("اسم المستخدم الجديد", "New Username"))
        password = st.text_input(_("كلمة المرور", "Password"), type="password")
        if st.form_submit_button(_("تسجيل", "Register")):
            if not username.strip() or not password.strip():
                st.error(_("الرجاء إدخال جميع البيانات", "Please enter all fields."))
                return
            if len(username) < 3 or len(password) < 5:
                st.error(_("اسم المستخدم يجب أن يكون 3 أحرف على الأقل وكلمة المرور 5 أحرف على الأقل",
                          "Username must be at least 3 chars and password at least 5 chars."))
                return
            if save_user(username, password):
                st.success(_("تم التسجيل بنجاح. يمكنك الآن تسجيل الدخول.", "Registered successfully. You can now log in."))
                st.session_state.page = "login"
                st.rerun()

def init_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "page" not in st.session_state:
        st.session_state.page = "login"

def main():
    init_session_state()
    st.markdown('<div style="background-color:#f0f2f6;padding:20px;border-radius:10px;">', unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #007BFF;'>{_('شات طبي', 'Medical Chatbot')}</h1>", unsafe_allow_html=True)
    if not st.session_state.logged_in:
        page_options = [_("تسجيل الدخول", "Login"), _("إنشاء حساب", "Sign Up")]
        page = st.sidebar.radio(_("اختر الإجراء", "Choose Action"), page_options)
        if page == page_options[0]:
            login_page()
        else:
            signup_page()
    else:
        chatbot_page()
        if st.sidebar.button(_("تسجيل الخروج", "Logout")):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

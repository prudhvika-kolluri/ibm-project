
import streamlit as st
import pandas as pd
import joblib
import html

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Poverty Support AI Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM DASHBOARD DESIGN
# ==========================================
st.markdown("""
<style>
    :root {
        --ink: #182b3a;
        --muted: #657887;
        --brand: #176b61;
        --line: #dce6e6;
        --surface: #ffffff;
    }
    .stApp { background: #f4f7f6; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244,247,246,.88); }
    [data-testid="stMainBlockContainer"] { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1380px; }
    [data-testid="stSidebar"] { background: #173b3b; border-right: 1px solid #28504d; }
    [data-testid="stSidebar"] * { color: #f4f8f6; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #b5cbc5; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .3rem .15rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap: .2rem; }
    .dashboard-title { color: var(--ink); font-size: clamp(1.8rem, 3vw, 2.55rem); line-height: 1.15; font-weight: 760; letter-spacing: -.035em; margin: .3rem 0 .55rem; }
    .dashboard-subtitle { color: var(--muted); font-size: 1rem; margin-bottom: 2rem; }
    .eyebrow { color: var(--brand); text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 750; }
    .metric-card { background: var(--surface); padding: 1.35rem 1.5rem; border-radius: 14px; border: 1px solid var(--line); min-height: 130px; box-shadow: 0 4px 15px rgba(23,59,59,.035); }
    .metric-label { color: var(--muted); font-size: .87rem; }
    .metric-value { color: var(--ink); font-size: 1.9rem; font-weight: 760; letter-spacing: -.03em; margin: .45rem 0 .2rem; }
    .scheme-card { background: var(--surface); padding: 1.35rem 1.5rem; border-radius: 14px; border: 1px solid var(--line); border-left: 4px solid var(--brand); margin: 1rem 0 .35rem; box-shadow: 0 4px 15px rgba(23,59,59,.035); }
    .scheme-name { font-size: 1.1rem; font-weight: 730; color: var(--ink); }
    h1, h2, h3 { color: var(--ink); letter-spacing: -.025em; }
    h2, h3 { margin-top: 1.25rem; }
    [data-testid="stForm"] { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 1.25rem 1.4rem .8rem; }
    div.stButton > button, div.stFormSubmitButton > button { border-radius: 8px; font-weight: 650; min-height: 2.8rem; transition: transform .15s ease, box-shadow .15s ease; }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] { background: var(--brand); border-color: var(--brand); }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover { transform: translateY(-1px); box-shadow: 0 5px 14px rgba(23,59,59,.14); }
    [data-testid="stAlert"] { border-radius: 10px; }
    [data-testid="stMetric"] { background: var(--surface); padding: 1rem 1.1rem; border: 1px solid var(--line); border-radius: 12px; }
    hr { border-color: var(--line); }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 2rem; }
        [data-testid="stForm"] { padding: .9rem; }
        .dashboard-subtitle { margin-bottom: 1.35rem; }
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("models/income_classifier.pkl")


try:
    model = load_model()
    model_error = None
except Exception as error:
    model = None
    model_error = str(error)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
# Process a requested page before creating the radio widget.
if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Dashboard"

if "requested_page" in st.session_state:
    st.session_state["navigation"] = st.session_state.pop(
        "requested_page"
    )

with st.sidebar:
    st.markdown("## 🏛️ Poverty Support AI")
    st.caption("Smart assistance dashboard")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Income Prediction",
            "Household Assessment",
            "Government Schemes",
            "About Project"
        ],
        key="navigation"
    )

    st.divider()
    st.caption("Academic project prototype")
    st.caption("AI + Government Scheme Matching")


# ==========================================
# COMMON HEADER
# ==========================================
st.markdown(
    '<div class="eyebrow">Community support • Information dashboard</div>'
    '<div class="dashboard-title">'
    'AI-Based Poverty Risk & Support Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Income analysis • Household needs • Government scheme discovery'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# DASHBOARD HOME PAGE
# ==========================================
if page == "Dashboard":

    st.subheader("Project Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Project Modules</div>
            <div class="metric-value">03</div>
            <div class="metric-label">
                Income, Household Needs and Schemes
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Schemes in Prototype</div>
            <div class="metric-value">03</div>
            <div class="metric-label">Preliminary screening rules</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model Status</div>
            <div class="metric-value">Demo</div>
            <div class="metric-label">US income dataset</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("### Welcome")

        st.write(
            "This project demonstrates how machine learning, "
            "household needs assessment, and government scheme "
            "discovery can be combined in one application."
        )

        st.info(
            "The current machine learning model uses US Census data. "
            "It is not a validated Indian poverty prediction model. "
            "Scheme results use separate, simplified screening rules."
        )

        st.markdown("### Available Features")
        st.markdown("""
        - Income category prediction
        - Household needs assessment
        - Preliminary government scheme matching
        - Links to official scheme information
        """)

    with right:
        st.markdown("### Quick Access")
        st.write("Open a project module:")

        if st.button(
            "Open Income Prediction",
            use_container_width=True
        ):
            st.session_state["requested_page"] = "Income Prediction"
            st.rerun()

        if st.button(
            "Assess Household Needs",
            use_container_width=True
        ):
            st.session_state["requested_page"] = "Household Assessment"
            st.rerun()

        if st.button(
            "Find Government Schemes",
            use_container_width=True
        ):
            st.session_state["requested_page"] = "Government Schemes"
            st.rerun()

    st.markdown("### Important Notice")
    st.warning(
        "This is an academic prototype. Results are informational only. "
        "They do not establish official poverty status or government "
        "scheme eligibility."
    )


# ==========================================
# INCOME PREDICTION PAGE
# ==========================================
elif page == "Income Prediction":

    st.subheader("Income Category Prediction")

    st.write(
        "Enter the person's details to run the existing machine "
        "learning model trained on US Census income data."
    )

    st.warning(
        "This model predicts a US income category. It must not be "
        "used to determine poverty status in India."
    )

    if model_error:
        st.error(f"Model could not be loaded: {model_error}")

    else:
        with st.form("income_form"):

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input(
                    "Age", 17, 100, 30
                )

                workclass = st.selectbox(
                    "Workclass",
                    [
                        "Private", "Self-emp-not-inc",
                        "Self-emp-inc", "Federal-gov",
                        "Local-gov", "State-gov",
                        "Without-pay", "Never-worked", "Unknown"
                    ]
                )

                education = st.selectbox(
                    "Education",
                    [
                        "Bachelors", "HS-grad", "Some-college",
                        "11th", "12th", "9th", "10th", "7th-8th",
                        "Masters", "Assoc-voc", "Assoc-acdm",
                        "Doctorate", "Prof-school", "5th-6th",
                        "1st-4th", "Preschool"
                    ]
                )

                education_num = st.number_input(
                    "Education number", 1, 16, 13
                )

                marital_status = st.selectbox(
                    "Marital status",
                    [
                        "Never-married", "Married-civ-spouse",
                        "Divorced", "Separated", "Widowed",
                        "Married-spouse-absent",
                        "Married-AF-spouse"
                    ]
                )

                occupation = st.selectbox(
                    "Occupation",
                    [
                        "Sales", "Craft-repair", "Exec-managerial",
                        "Adm-clerical", "Other-service",
                        "Machine-op-inspct", "Prof-specialty",
                        "Transport-moving", "Handlers-cleaners",
                        "Farming-fishing", "Tech-support",
                        "Protective-serv", "Priv-house-serv",
                        "Armed-Forces", "Unknown"
                    ]
                )

                relationship = st.selectbox(
                    "Relationship",
                    [
                        "Not-in-family", "Husband", "Wife",
                        "Own-child", "Unmarried", "Other-relative"
                    ]
                )

            with col2:
                fnlwgt = st.number_input(
                    "Final weight (fnlwgt)",
                    min_value=1,
                    value=150000
                )

                race = st.selectbox(
                    "Race",
                    [
                        "White", "Black", "Asian-Pac-Islander",
                        "Amer-Indian-Eskimo", "Other"
                    ]
                )

                sex = st.selectbox(
                    "Sex", ["Female", "Male"]
                )

                capital_gain = st.number_input(
                    "Capital gain", min_value=0, value=0
                )

                capital_loss = st.number_input(
                    "Capital loss", min_value=0, value=0
                )

                hours_per_week = st.number_input(
                    "Hours worked per week", 1, 100, 40
                )

                native_country = st.selectbox(
                    "Native country",
                    [
                        "United-States", "India", "Mexico",
                        "Cuba", "Canada", "England", "Germany", "Other"
                    ]
                )

            predict = st.form_submit_button(
                "Predict Income Category",
                use_container_width=True
            )

        if predict:
            person = pd.DataFrame([{
                "age": age,
                "workclass": workclass,
                "fnlwgt": fnlwgt,
                "education": education,
                "education_num": education_num,
                "marital_status": marital_status,
                "occupation": occupation,
                "relationship": relationship,
                "race": race,
                "sex": sex,
                "capital_gain": capital_gain,
                "capital_loss": capital_loss,
                "hours_per_week": hours_per_week,
                "native_country": native_country
            }])

            try:
                result = model.predict(person)[0]

                st.markdown("### Prediction Result")

                if result == "<=50K":
                    st.warning(
                        "Predicted category: $50,000 or less per year"
                    )
                else:
                    st.success(
                        "Predicted category: Above $50,000 per year"
                    )

                st.caption(
                    "This is a US income-category estimate, not an "
                    "Indian poverty assessment."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# ==========================================
# HOUSEHOLD ASSESSMENT PAGE
# ==========================================
elif page == "Household Assessment":

    st.subheader("Household Needs Assessment")

    st.write(
        "Answer these questions about household living conditions. "
        "The results highlight reported needs; they do not calculate "
        "an official poverty score."
    )

    questions = {
        "Nutritious food": (
            "Does the household have enough nutritious food every day?"
        ),
        "Safe drinking water": (
            "Does the household have access to safe drinking water?"
        ),
        "Sanitation": (
            "Does the household have access to a suitable toilet?"
        ),
        "Electricity": (
            "Does the household have reliable electricity?"
        ),
        "Clean cooking fuel": (
            "Does the household have access to clean cooking fuel?"
        ),
        "Suitable housing": (
            "Does the household have safe and suitable housing?"
        ),
        "School attendance": (
            "Do all school-aged children in the household attend school?"
        ),
        "Healthcare access": (
            "Can household members access healthcare when needed?"
        ),
        "Health coverage": (
            "Does at least one household member have health coverage?"
        ),
        "Bank account": (
            "Does the household have access to a bank account?"
        )
    }

    with st.form("household_assessment_form"):

        responses = {}

        for key, question in questions.items():
            responses[key] = st.radio(
                question,
                ["Yes", "No", "Not sure"],
                horizontal=True,
                key=f"assessment_{key}"
            )

        assess_clicked = st.form_submit_button(
            "Assess Household Needs",
            use_container_width=True
        )

    if assess_clicked:

        needs = [
            name for name, answer in responses.items()
            if answer == "No"
        ]

        uncertain = [
            name for name, answer in responses.items()
            if answer == "Not sure"
        ]

        st.markdown("### Assessment Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Reported needs", len(needs))

        with col2:
            st.metric(
                "Answers needing clarification",
                len(uncertain)
            )

        if needs:
            st.markdown("#### Areas that may need support")

            for need in needs:
                st.write(f"- {need}")

        else:
            st.success(
                "No unmet needs were reported in the areas assessed."
            )

        if uncertain:
            st.markdown("#### Information to clarify")

            for item in uncertain:
                st.write(f"- {item}")

        st.info(
            "This is a preliminary checklist, not an official "
            "Multidimensional Poverty Index calculation or a "
            "government eligibility decision."
        )

        st.markdown(
            "[Explore government schemes on myScheme]"
            "(https://www.myscheme.gov.in/)"
        )


# ==========================================
# GOVERNMENT SCHEME FINDER PAGE
# ==========================================
elif page == "Government Schemes":

    st.subheader("Indian Government Scheme Finder")

    st.write(
        "Enter the applicant's details to find preliminary scheme "
        "matches among the three schemes included in this prototype."
    )

    with st.form("scheme_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Applicant age", 1, 120, 25
            )

            income = st.number_input(
                "Annual family income (₹)",
                min_value=0,
                value=100000,
                step=5000
            )

            category = st.selectbox(
                "Social category",
                ["SC", "ST", "OBC", "EBC", "EWS", "DNT", "Other"]
            )

            safai_mitra = st.selectbox(
                "Safai Mitra?",
                ["No", "Yes"]
            )

            student = st.selectbox(
                "Studying beyond Class XII?",
                ["No", "Yes"]
            )

        with col2:
            notified = st.selectbox(
                "Admitted to a notified institution for PM-YASASVI?",
                ["No", "Yes", "Not sure"]
            )

            siblings = st.selectbox(
                "Siblings receiving the same PM-YASASVI scholarship?",
                ["0", "1", "2 or more", "Not sure"]
            )

            trade = st.selectbox(
                "Practising one of the 18 PM Vishwakarma trades?",
                ["No", "Yes"]
            )

            artisan = st.selectbox(
                "Self-employed artisan working with hands and tools?",
                ["No", "Yes"]
            )

            loan = st.selectbox(
                "Similar government business loan in past 5 years?",
                ["No", "Yes", "Not sure"]
            )

            govt_service = st.selectbox(
                "Applicant or family member in government service?",
                ["No", "Yes"]
            )

        find = st.form_submit_button(
            "Find Potential Matches",
            use_container_width=True
        )

    if find:
        matches = []

        # PM-DAKSH: simplified demonstration rules only.
        daksh_category = (
            category in ["SC", "OBC", "EWS", "DNT"]
            or safai_mitra == "Yes"
        )

        daksh_income = (
            category in ["SC", "DNT"]
            or safai_mitra == "Yes"
            or (
                category in ["OBC", "EWS"]
                and income < 300000
            )
        )

        if 18 <= age <= 45 and daksh_category and daksh_income:
            matches.append({
                "name": "PM-DAKSH",
                "benefit": (
                    "Skill training and employment-oriented courses"
                ),
                "url": "https://www.myscheme.gov.in/schemes/pm-daksh"
            })

        # PM Vishwakarma: simplified demonstration rules only.
        if (
            age >= 18
            and trade == "Yes"
            and artisan == "Yes"
            and loan == "No"
            and govt_service == "No"
        ):
            matches.append({
                "name": "PM Vishwakarma",
                "benefit": (
                    "Training, toolkit incentive and other support"
                ),
                "url": "https://www.myscheme.gov.in/schemes/pmv"
            })

        # PM-YASASVI: simplified demonstration rules only.
        if (
            category in ["OBC", "EBC", "DNT"]
            and income <= 250000
            and student == "Yes"
            and notified == "Yes"
            and siblings in ["0", "1"]
        ):
            matches.append({
                "name": "PM-YASASVI: Top Class College Education",
                "benefit": (
                    "Financial support for eligible students, "
                    "subject to the applicable scheme rules"
                ),
                "url": (
                    "https://www.myscheme.gov.in/schemes/"
                    "pm-yasasvitcceobcebcdnts"
                )
            })

        st.markdown("### Recommendation Results")

        a, b = st.columns(2)

        with a:
            st.metric("Potential matches", len(matches))

        with b:
            st.metric("Schemes checked", 3)

        if matches:
            for scheme in matches:
                safe_name = html.escape(scheme["name"])
                safe_benefit = html.escape(scheme["benefit"])

                st.markdown(
                    f"""
                    <div class="scheme-card">
                        <div class="scheme-name">{safe_name}</div>
                        <p>{safe_benefit}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"[View official scheme details]({scheme['url']})"
                )

        else:
            st.info(
                "No potential matches were found among these three "
                "schemes. Other schemes may still be available."
            )

        st.warning(
            "Preliminary screening only. These simplified rules do not "
            "cover every official condition and do not verify documents. "
            "A match is not confirmation of eligibility."
        )

    st.markdown("---")

    st.markdown(
        "Explore more schemes on the "
        "[official Government of India myScheme portal]"
        "(https://www.myscheme.gov.in/)."
    )


# ==========================================
# ABOUT PROJECT PAGE
# ==========================================
elif page == "About Project":

    st.subheader("About the Project")

    st.write(
        "This academic prototype demonstrates how machine learning, "
        "a household needs checklist, and rule-based scheme matching "
        "can help users explore possible support options."
    )

    st.markdown("### Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Joblib
    """)

    st.markdown("### Current Limitations")

    st.markdown("""
    - The trained income model uses a US Census dataset.
    - It is not a validated Indian poverty prediction model.
    - Household assessment is a preliminary checklist.
    - Scheme matching uses simplified demonstration rules.
    - Official eligibility and documentation must be verified.
    """)

    st.markdown("### Official Scheme Resource")

    st.markdown(
        "[Government of India myScheme portal]"
        "(https://www.myscheme.gov.in/)"
    )

    st.caption(
        "Future improvement: use suitable Indian household data and "
        "verified scheme eligibility rules."
    )

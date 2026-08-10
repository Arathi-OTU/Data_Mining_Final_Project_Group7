
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from openai import OpenAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Amazon Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide"
)


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_1.pkl")


vectorizer = load_vectorizer()


# ============================================================
# LOAD MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource
def load_models():

    logistic_model = joblib.load(
        "sentiment_model_lr_modified.pkl"
    )

    naive_bayes_model = joblib.load(
        "sentiment_model_nb.pkl"
    )

    linear_svc_model = joblib.load(
        "sentiment_model_svc.pkl"
    )

    return {
        "Logistic Regression": logistic_model,
        "Multinomial Naïve Bayes": naive_bayes_model,
        "Linear SVC": linear_svc_model
    }


models = load_models()


# ============================================================
# OPENAI CONFIGURATION
# ============================================================

# Change this if you want to use another OpenAI model.
OPENAI_MODEL = "gpt-5"


# ============================================================
# INITIALIZE OPENAI CLIENT
# ============================================================

openai_client = None

try:

    if "OPENAI_API_KEY" in st.secrets:

        openai_client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )

except Exception:

    openai_client = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_sentiment(prediction):
    """
    Convert a model prediction into:
    Positive / Negative / Neutral
    """

    prediction = str(prediction).lower().strip()

    if "positive" in prediction:
        return "Positive"

    elif "negative" in prediction:
        return "Negative"

    elif "neutral" in prediction:
        return "Neutral"

    return "Unknown"


def sentiment_emoji(sentiment):

    if sentiment == "Positive":
        return "😊"

    elif sentiment == "Negative":
        return "😞"

    elif sentiment == "Neutral":
        return "😐"

    return "❓"


def display_sentiment(sentiment):

    emoji = sentiment_emoji(sentiment)

    if sentiment == "Positive":

        st.success(
            f"{emoji} {sentiment}"
        )

    elif sentiment == "Negative":

        st.error(
            f"{emoji} {sentiment}"
        )

    elif sentiment == "Neutral":

        st.info(
            f"{emoji} {sentiment}"
        )

    else:

        st.warning(
            f"{emoji} {sentiment}"
        )


# ============================================================
# OPENAI SENTIMENT CLASSIFICATION
# ============================================================

def get_openai_sentiment(review):

    if openai_client is None:

        return None

    prompt = f"""
Classify the sentiment of the following Amazon product review.

Choose exactly ONE of these labels:

Positive
Negative
Neutral

Definitions:

Positive:
The customer is satisfied, happy, or expresses favorable
opinions about the product.

Negative:
The customer is dissatisfied, unhappy, or expresses unfavorable
opinions about the product.

Neutral:
The review is mostly factual, mixed, or does not clearly express
positive or negative sentiment.

Return ONLY one label:
Positive
Negative
Neutral

Do not provide an explanation.

Review:
{review}
"""

    try:

        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            instructions=(
                "You are a sentiment classification system. "
                "Return exactly one label: Positive, Negative, or Neutral."
            ),
            input=prompt
        )

        result = response.output_text.strip()

        return normalize_sentiment(result)

    except Exception as e:

        st.error(
            f"OpenAI classification error: {e}"
        )

        return None


# ============================================================
# ML PREDICTION
# ============================================================

def get_ml_prediction(model, review_vector):

    prediction = model.predict(
        review_vector
    )[0]

    return normalize_sentiment(
        prediction
    )


# ============================================================
# CONFIDENCE CALCULATION
# ============================================================

def calculate_confidence(model, review_vector):

    confidence = None

    # --------------------------------------------------------
    # Logistic Regression / Naïve Bayes
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            review_vector
        )[0]

        confidence = (
            np.max(probabilities) * 100
        )

    # --------------------------------------------------------
    # Linear SVC
    # --------------------------------------------------------

    elif hasattr(model, "decision_function"):

        decision_scores = model.decision_function(
            review_vector
        )

        if len(decision_scores.shape) == 1:

            score = float(
                decision_scores[0]
            )

            confidence = (
                1 /
                (
                    1 +
                    np.exp(-abs(score))
                )
            ) * 100

        else:

            scores = np.asarray(
                decision_scores
            )

            exp_scores = np.exp(
                scores - np.max(scores)
            )

            probabilities = (
                exp_scores /
                exp_scores.sum()
            )

            confidence = (
                np.max(probabilities) * 100
            )

    return confidence


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🛍️ Amazon Review Sentiment Analyzer
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Machine Learning and OpenAI Sentiment Classification
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Model Settings"
)


selected_model = st.sidebar.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "Multinomial Naïve Bayes",
        "Linear SVC"
    ]
)


st.sidebar.divider()


st.sidebar.subheader(
    "Available ML Models"
)

st.sidebar.write(
    "🔵 Logistic Regression"
)

st.sidebar.write(
    "🟢 Multinomial Naïve Bayes"
)

st.sidebar.write(
    "🟠 Linear SVC"
)


st.sidebar.divider()


st.sidebar.info(
    "All machine learning models use TF-IDF "
    "features generated from the review text."
)


# ============================================================
# OPENAI SIDEBAR
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "🧠 OpenAI"
)

st.sidebar.write(
    f"Model: `{OPENAI_MODEL}`"
)


if openai_client is not None:

    st.sidebar.success(
        "OpenAI API connected"
    )

else:

    st.sidebar.warning(
        "OpenAI API key not configured"
    )


# ============================================================
# REVIEW INPUT METHOD
# ============================================================

st.subheader(
    "📝 Review Input"
)

input_method = st.radio(
    "Choose how you want to provide the review:",
    [
        "✍️ Enter Review Manually",
        "💡 Choose Sample Review"
    ],
    horizontal=True
)


# ============================================================
# MANUAL REVIEW INPUT
# ============================================================

review = ""


if input_method == "✍️ Enter Review Manually":

    review = st.text_area(
        "Enter your Amazon review:",
        placeholder=(
            "Example: The product is excellent! "
            "The quality is great and delivery was very fast."
        ),
        height=180
    )


# ============================================================
# SAMPLE REVIEW INPUT
# ============================================================

else:

    sample_review = st.selectbox(
        "Choose a sample Amazon review:",
        [
            "Select an example...",

            "The product is excellent and works perfectly!",

            "The product stopped working after two days. "
            "Very disappointing.",

            "The product is okay. Nothing special.",

            "I absolutely love this product. "
            "The quality is amazing and I would definitely buy it again.",

            "The item arrived damaged and does not work. "
            "Very poor quality.",

            "The product arrived on time. "
            "It works as expected."
        ]
    )


    if sample_review != "Select an example...":

        review = sample_review

        st.info(
            f"**Selected Review:** {review}"
        )


st.divider()


# ============================================================
# PREDICT SENTIMENT
# ============================================================

if st.button(
    "🔍 Predict Sentiment",
    use_container_width=True,
    type="primary"
):

    if review.strip() == "":

        st.warning(
            "Please enter a review or select a sample review."
        )

    else:

        # ====================================================
        # SELECT ML MODEL
        # ====================================================

        model = models[
            selected_model
        ]


        # ====================================================
        # TF-IDF TRANSFORMATION
        # ====================================================

        review_vector = vectorizer.transform(
            [review]
        )


        # ====================================================
        # ML PREDICTION
        # ====================================================

        ml_sentiment = get_ml_prediction(
            model,
            review_vector
        )


        # ====================================================
        # OPENAI PREDICTION
        # ====================================================

        with st.spinner(
            "🧠 OpenAI is analyzing the review..."
        ):

            openai_sentiment = (
                get_openai_sentiment(
                    review
                )
            )


        # ====================================================
        # SENTIMENT COMPARISON
        # ====================================================

        st.subheader(
            "🎯 Sentiment Classification Comparison"
        )


        result_col1, result_col2 = st.columns(
            2
        )


        # ----------------------------------------------------
        # ML RESULT
        # ----------------------------------------------------

        with result_col1:

            st.markdown(
                "### 🤖 Machine Learning"
            )

            display_sentiment(
                ml_sentiment
            )

            st.caption(
                f"Model: {selected_model}"
            )


        # ----------------------------------------------------
        # OPENAI RESULT
        # ----------------------------------------------------

        with result_col2:

            st.markdown(
                "### 🧠 OpenAI"
            )

            if openai_sentiment is not None:

                display_sentiment(
                    openai_sentiment
                )

                st.caption(
                    f"Model: {OPENAI_MODEL}"
                )

            else:

                st.warning(
                    "OpenAI prediction unavailable."
                )


        # ====================================================
        # AGREEMENT
        # ====================================================

        st.divider()

        st.subheader(
            "🔎 Classification Agreement"
        )


        if openai_sentiment is not None:

            if ml_sentiment == openai_sentiment:

                st.success(
                    f"✅ Both classifiers agree: "
                    f"**{ml_sentiment}**"
                )

            else:

                st.warning(
                    f"⚠️ The classifiers disagree. "
                    f"ML: **{ml_sentiment}** | "
                    f"OpenAI: **{openai_sentiment}**"
                )

        else:

            st.info(
                "Agreement cannot be calculated because "
                "the OpenAI prediction is unavailable."
            )


        # ====================================================
        # COMPARISON TABLE
        # ====================================================

        st.subheader(
            "📋 Classification Comparison Table"
        )


        comparison_data = [
            {
                "Classifier": "Machine Learning",
                "Model": selected_model,
                "Sentiment": ml_sentiment
            }
        ]


        if openai_sentiment is not None:

            comparison_data.append(
                {
                    "Classifier": "OpenAI",
                    "Model": OPENAI_MODEL,
                    "Sentiment": openai_sentiment
                }
            )


        comparison_df = pd.DataFrame(
            comparison_data
        )


        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # ML CONFIDENCE
        # ====================================================

        st.subheader(
            "📊 ML Prediction Confidence"
        )


        confidence = calculate_confidence(
            model,
            review_vector
        )


        if confidence is not None:

            st.progress(
                min(
                    confidence / 100,
                    1.0
                )
            )

            st.write(
                f"**Confidence: "
                f"{confidence:.2f}%**"
            )

        else:

            st.info(
                "Confidence score is not available "
                "for this model."
            )


        # ====================================================
        # ANALYZED REVIEW
        # ====================================================

        st.subheader(
            "📝 Analyzed Review"
        )

        st.info(
            review
        )


        # ====================================================
        # IMPORTANT TF-IDF WORDS
        # ====================================================

        st.subheader(
            "🔑 Important Words"
        )


        feature_names = (
            vectorizer
            .get_feature_names_out()
        )


        tfidf_values = (
            review_vector
            .toarray()[0]
        )


        top_indices = (
            tfidf_values
            .argsort()[-10:][::-1]
        )


        important_words = []


        for index in top_indices:

            if tfidf_values[index] > 0:

                important_words.append(
                    feature_names[index]
                )


        if important_words:

            st.write(
                " • ".join(
                    important_words
                )
            )

        else:

            st.write(
                "No significant TF-IDF words found."
            )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()


st.subheader(
    "🔬 Compare All Three ML Models with OpenAI"
)


st.write(
    "Run the same review through all three machine learning "
    "classifiers and compare their predictions with OpenAI."
)


if st.button(
    "⚡ Compare Models",
    use_container_width=True
):

    if review.strip() == "":

        st.warning(
            "Please enter a review or select a sample review."
        )

    else:

        # ====================================================
        # TF-IDF
        # ====================================================

        review_vector = vectorizer.transform(
            [review]
        )


        # ====================================================
        # OPENAI PREDICTION
        # ====================================================

        with st.spinner(
            "🧠 Getting OpenAI classification..."
        ):

            openai_comparison_sentiment = (
                get_openai_sentiment(
                    review
                )
            )


        # ====================================================
        # COMPARE ALL ML MODELS
        # ====================================================

        comparison_results = []


        for model_name, model in models.items():

            prediction = get_ml_prediction(
                model,
                review_vector
            )


            if openai_comparison_sentiment is not None:

                agreement = (
                    "✅ Yes"
                    if prediction ==
                    openai_comparison_sentiment
                    else "❌ No"
                )

            else:

                agreement = "N/A"


            comparison_results.append(
                {
                    "Model": model_name,

                    "ML Prediction": (
                        f"{sentiment_emoji(prediction)} "
                        f"{prediction}"
                    ),

                    "OpenAI Prediction": (
                        f"{sentiment_emoji(openai_comparison_sentiment)} "
                        f"{openai_comparison_sentiment}"
                        if openai_comparison_sentiment
                        is not None
                        else "Unavailable"
                    ),

                    "Agreement": agreement
                }
            )


        comparison_df = pd.DataFrame(
            comparison_results
        )


        # ====================================================
        # DISPLAY TABLE
        # ====================================================

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # AGREEMENT SUMMARY
        # ====================================================

        if openai_comparison_sentiment is not None:

            agreement_count = sum(
                1
                for row in comparison_results
                if row["Agreement"] == "✅ Yes"
            )


            total_models = len(
                comparison_results
            )


            agreement_percentage = (
                agreement_count /
                total_models
            ) * 100


            st.subheader(
                "📈 Agreement Summary"
            )


            summary_col1, summary_col2, summary_col3 = (
                st.columns(3)
            )


            with summary_col1:

                st.metric(
                    "ML Models",
                    total_models
                )


            with summary_col2:

                st.metric(
                    "Agreements",
                    agreement_count
                )


            with summary_col3:

                st.metric(
                    "Agreement Rate",
                    f"{agreement_percentage:.1f}%"
                )


        else:

            st.warning(
                "OpenAI prediction was unavailable, "
                "so agreement statistics could not be calculated."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Group 7 | Amazon Product Review Sentiment Analysis | "
    "TF-IDF + Machine Learning + OpenAI"
)


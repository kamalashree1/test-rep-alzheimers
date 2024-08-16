import streamlit as st
from openai import OpenAI
import numpy as np
import pickle
import json
import requests


# don't change this constant: Rootname
ROOT_NAME = "Shilpi"
MODEL_NAME = "full_transcript_mlp_best_model"
MAPPER = {
    "0": "Normal",
    "1": "Cognitive Dementia"
}


# create client
client = OpenAI(
    api_key = st.secrets["OPENAI_API_KEY"]
)


def save_feedbacks(data):
  url = 'https://s5bcsbu84l.execute-api.us-east-1.amazonaws.com/Research/record-history'
  r = requests.post(url, data=json.dumps(data))
  response = getattr(r,'_content').decode("utf-8")

  response = json.loads(response)
  print(response)
  saved = response["data"]["saved"]
  return saved


def create_payload_to_record(user_id, user_context, label):
    payload = {
        "httpMethod": "POST",
        "body": {
            "rootName": ROOT_NAME,
            "userId": user_id,
            "userText": user_context,
            "prediction": label
        }
    }

    return payload


@st.cache_resource
def load_model(path: str):
    with open(path, 'rb') as pickle_file:
        model_file = pickle.load(pickle_file)

    return model_file


def get_openai_embeddings(text: str) -> list:
    response = client.embeddings.create(
        input = f"{text}",
        model = "text-embedding-ada-002"
    )

    return response.data[0].embedding


def dementia_app():
    # get query params
    authorization_response = st.query_params

    final_model = load_model(MODEL_NAME)

    if "code" in authorization_response:
        # set the title
        st.header("Demential Tracker")

        # enter the text input
        text_input = st.text_input("Please Enter the Text")

        if text_input:
            if st.button("Make Predictions"):
                # get the embeddings
                embeddings = get_openai_embeddings(text_input)
                # predictions
                pred = final_model.predict(np.array([embeddings]))
                pred_label = MAPPER[str(pred[0])]
                # record the results
                set_payload = create_payload_to_record(st.session_state.user_id, text_input, pred_label)
                saved_db = save_feedbacks(set_payload)
                if saved_db:
                    st.toast("Record Saved!")
                    st.subheader("Your Condition is {}".format(pred_label))
                else:
                    st.error("Saving Records Failed!")
                
    else:
        st.error("Please Sign In First!")

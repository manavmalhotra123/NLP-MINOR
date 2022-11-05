# Core packages 
import streamlit
import altair as alt

# EDA packages
import pandas as pd
import numpy as np

# util package 
import joblib

 
pipe_lr = joblib.load(open("Model_trained.pkl","rb"))
# function
def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_prob(docx):
    results = pipe_lr.predict_proba([docx])
    return results
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


def main():
    streamlit.title("Emotion Classifier Engine")
    menu = ["Home","Monitor","About"]

    choice = streamlit.sidebar.selectbox('Menu',menu)

    if choice == "Home":
        streamlit.subheader("Home-Emotion In Text")

        with streamlit.form(key='Emotion_clf_form'):
            raw_text = streamlit.text_area('Type your Text Here')
            submit = streamlit.form_submit_button(label="Submit")
        
        if submit:
            col1,col2 = streamlit.beta_columns(2)
            
            # applying the function here 
            prediction = predict_emotion(raw_text)
            probability = get_prediction_prob(raw_text)
            with col1:
                streamlit.success("Original Text")
                streamlit.write(raw_text)
                streamlit.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                streamlit.write("{}:{}".format(prediction,emoji_icon))
                streamlit.write("confidence:{}".format(np.max(probability)))           
            
            with col2:
                streamlit.success("Prediction Probability ")
                #streamlit.write(prediction)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                #streamlit.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x="Emotions",y="Probability",color="Emotions")
                streamlit.altair_chart(fig,use_container_width=True)

    elif choice == "Monitor":
        streamlit.subheader("Monitor App")
    
    else:
        streamlit.subheader("About")
    

if __name__ == "__main__":
    main()
#######################################################
import os, datetime, json, sys, pathlib, shutil
import pandas as pd
import streamlit as st
import cv2
import face_recognition
import numpy as np
from settings import VISITOR_HISTORY, VISITOR_DB
#######################################################
## Disable Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)
################################################### Defining Static Data ###############################################
st.sidebar.image('https://i.flockusercontent2.com/q884s08qs8s9s4lb?r=1157408321',
                 use_column_width=False)
st.sidebar.markdown("""
    > Made by [*Ashish Gopal*](https://www.linkedin.com/in/ashish-gopal)
    """)

user_color      = '#000000'
title_webapp    = "Visitor Monitoring Webapp"

html_temp = f"""
            <div style="background-color:{user_color};padding:12px">
            <h1 style="color:white;text-align:center;">{title_webapp}
            <img src = ""
            align="right" width=230px ></h1>
            </div>
            """
st.markdown(html_temp, unsafe_allow_html=True)

###################### Defining Static Paths ###################4
if st.sidebar.checkbox('Click to Clear out all the data'):
    shutil.rmtree(VISITOR_DB, ignore_errors=True)
    os.mkdir(VISITOR_DB)

    shutil.rmtree(VISITOR_HISTORY, ignore_errors=True)
    VISITOR_HISTORY.mkdir()


if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)

if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)
# st.write(VISITOR_HISTORY)
################################################### Parameters #####################################################
COLOR_DARK  = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO   = ['name']
COLS_ENCODE = [f'v{i}' for i in range(128)]

## Database
data_path   = VISITOR_DB
file_name   = 'visitors_db.csv'

# images      = []
# classNames  = []
#
# myList      = [file for file in os.listdir(VISITOR_DB) if file.endswith('.jpg')]
# # st.write(myList)
#
# for cl in myList:
#     curImg = cv2.imread(f'{VISITOR_DB}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# # st.write(classNames)

################################################### Defining Function ##############################################
def init_data():
    if os.path.exists(os.path.join(data_path, file_name)):
        # st.info('Database Found!')
        # st.write(os.path.join(data_path, file_name))
        df = pd.read_csv(os.path.join(data_path, file_name))

    else:
        # st.info('Database Not Found!')
        df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
        df.to_csv(os.path.join(data_path, file_name), index=False)

    return df

def add_data(df_visitor_details):
    try:
        df_all = pd.read_csv(os.path.join(data_path, file_name))

        if not df_all.empty:
            df_all = df_all.append(df_visitor_details, ignore_index=False)
            df_all.drop_duplicates().reset_index(inplace=True, drop=True)
            df_all.to_csv(os.path.join(data_path, file_name), index=False)
            st.success('Details Added Successfully!')
        else:
            df_visitor_details.to_csv(os.path.join(data_path, file_name), index=False)
            st.success('Initiated Data Successfully!')

    except Exception as e:
        st.error(e)

# convert opencv BRG to regular RGB mode
def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

def findEncodings(images):
    encode_list = []

    for img in images:
        img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode  = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))

def attendace(name):
    f_p = os.path.join(VISITOR_DB, 'attendance.csv')
    # st.write(f_p)

    now         = datetime.datetime.now()
    dtString    = now.strftime('%H:%M:%S')
    df_attendace_temp = pd.DataFrame(data={
                                            "visitor_name": [name],
                                            "Timing": [dtString]
                                        })

    if not os.path.isfile(f_p):
        df_attendace_temp.to_csv(f_p, index=False)
        st.write(df_attendace_temp)
    else:
        df_attendace = pd.read_csv(f_p)
        df_attendace = df_attendace.append(df_attendace_temp)
        df_attendace.to_csv(f_p, index=False)

        st.write(df_attendace)

#######################################################
def main():
    st.sidebar.header("About")
    st.sidebar.info("This webapp gives a demo of Visitor Monitoring Webapp using 'Face Recognition' using Streamlit")
    ###################################################

    # encodeListKnown = findEncodings(images)

    ###################################################
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        # convert image from opened file to np.array
        image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # st.image(cv2_img)

        with open(os.path.join(VISITOR_HISTORY, 'visitor.jpg'), 'wb') as file:
            file.write(img_file_buffer.getbuffer())
            st.success('Image Saved Successfully!')
            flag_input = True

            ## Validate Image
            # detect faces in the loaded image
            max_faces   = 0
            rois        = []  # region of interests (arrays of face areas)

            face_locations  = face_recognition.face_locations(image_array)
            encodesCurFrame = face_recognition.face_encodings(image_array, face_locations)

            for idx, (top, right, bottom, left) in enumerate(face_locations):
                # save face region of interest to list
                rois.append(image_array[top:bottom, left:right].copy())

                # Draw a box around the face and lable it
                cv2.rectangle(image_array, (left, top),(right, bottom), COLOR_DARK, 2)
                cv2.rectangle(image_array, (left, bottom + 35),(right, bottom), COLOR_DARK, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image_array, f"#{idx}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

            st.image(BGR_to_RGB(image_array), width=720)
            max_faces = len(face_locations)

            if max_faces > 0:
                col1, col2 = st.columns(2)

                # select interested face in picture
                face_idx = col1.selectbox("Select face#", range(max_faces))

                if col1.checkbox('Click to proceed!'):
                    roi = rois[face_idx]
                    st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

                    # initial database for known faces
                    DB = init_data()
                    # st.write(DB)

                    face_encodings  = DB[COLS_ENCODE].values
                    dataframe       = DB[COLS_INFO]

                    # compare roi to known faces, show distances and similarities
                    faces = face_recognition.face_encodings(roi)

                    if len(faces)<1:
                        st.error('Please Try Again!')
                    else:
                        face_to_compare = faces[0]
                        dataframe['distance']   = face_recognition.face_distance(face_encodings, face_to_compare)
                        dataframe['distance'] = dataframe['distance'].astype(float)

                        dataframe['similarity'] = dataframe.distance.apply(lambda distance: f"{face_distance_to_conf(distance):0.2}")
                        dataframe['similarity'] = dataframe['similarity'].astype(float)

                        dataframe_new   = dataframe.drop_duplicates(keep='first')
                        dataframe_new.reset_index(drop=True, inplace=True)
                        dataframe_new.sort_values(by="similarity", ascending=True)

                        ## Filtering
                        dataframe_new = dataframe_new[dataframe_new['similarity']>0.9].head(1)
                        dataframe_new.reset_index(drop=True, inplace=True)

                        with st.expander('Click to see Matching logs'):
                            st.write(dataframe_new)
                            # st.write(dataframe_new.dtypes)
                        ########################
                        if dataframe_new.empty:
                            name_visitor = st.text_input('Enter Name of new visitor')

                            with open(os.path.join(VISITOR_HISTORY, f'{name_visitor}.jpg'), 'wb') as file:
                                file.write(img_file_buffer.getbuffer())
                                st.success('Image Saved Successfully!')
                        else:
                            name_visitor = dataframe_new['name'].loc[0]

                        with st.expander('Click to see Visitor logs'):
                            attendace(name_visitor)

                        # add Image to database
                        if st.checkbox('Add to Database'):
                            face_name = st.text_input('Name:', '')
                            face_des = st.text_input('Desciption:', '')
                            if st.button('Add to Database'):
                                encoding = face_to_compare.tolist()
                                # st.write(DB)
                                DB.loc[len(DB)] = [face_name, face_des] + encoding
                                # st.write(DB)
                                add_data(DB)



                                with open(os.path.join(VISITOR_DB, f'{face_name}.jpg'), 'wb') as file:
                                    file.write(img_file_buffer.getbuffer())
                                    st.success('Image Saved Successfully!')


            else:
                st.write('No human face detected.')
#######################################################
if __name__ == "__main__":
    main()
#######################################################

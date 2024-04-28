# import necessary libraries here
import cv2
import streamlit as st
from ultralytics import YOLO


def app():
    # Add app code here
    model = YOLO("yolov8n.pt")
    objectNames = list(model.names.values())
    header = st.header("Object Detection Web Application")
    subHeader = st.subheader("Powered by YoloV8")
    st.write("Welcome!")
    st.image("img.png")
    with st.form("my_form"):
        uploadFile = st.file_uploader("Upload video", type=['mp4'])

        selectedObjects = st.multiselect('Choose objects to detect', objectNames, default=['person'])
        thresholdValue = st.slider(label='Confidence Score', min_value=0.0,
                                   max_value=1.0)
        st.form_submit_button(label='Submit')

    if uploadFile is not None:
        inputPath = uploadFile.name
        fileBinary = uploadFile.read()
        with open(inputPath, "wb") as tempFile:
            tempFile.write(fileBinary)
        videoStream = cv2.VideoCapture(inputPath)
        width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'h264')
        fps = int(videoStream.get(cv2.CAP_PROP_FPS))
        output_path = inputPath.split('.')[0] + '_output.mp4'
        outVideo = cv2.VideoWriter(output_path, int(fourcc), fps, (width, height))

        with st.spinner("Processing Video ... "):
            while True:
                ret, frame = videoStream.read()
                if not ret:
                    break
                result = model(frame)
                for detection in result[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    resultClass = int(detection[5])
                    objectName = model.names[resultClass]
                    label = f'{objectName} {score}'

                    if model.names[resultClass] in selectedObjects and score > thresholdValue:
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                detections = result[0].verbose()
                cv2.putText(frame, detections, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                outVideo.write(frame)
            videoStream.release()
            outVideo.release()
        st.video(output_path)


if __name__ == "__main__":
    app()
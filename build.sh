version=v1.0.0.`date +%s` && \
docker build . -t bianchx/face_recognition_api:$version && \
docker tag bianchx/face_recognition_api:$version bianchx/face_recognition_api && \
docker push bianchx/face_recognition_api:$version && \
docker push bianchx/face_recognition_api
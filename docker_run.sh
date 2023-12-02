docker build -t train_model .

docker run -v "$()/app/models" model_train
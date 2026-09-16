import mlflow

# Укажите ваш run_id зависшего запуска
run_id = "8a34a3a204a04bfeafc3a4f67ce746e9"

client = mlflow.tracking.MlflowClient(tracking_uri="sqlite:///mlflow.db")
client.set_terminated(run_id, status="FINISHED")  # или "FAILED"
print(f"Запуск {run_id} принудительно закрыт.")

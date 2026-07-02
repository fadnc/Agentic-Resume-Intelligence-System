# stop-all.sh
for svc in dis-v7-backend dis-v7-frontend dis-v7-mlflow dis-v7-grafana; do
  aws ecs update-service --cluster dis-v7-cluster --service $svc --desired-count 0 --region ap-south-1
  echo "Stopped $svc"
done
echo "All services stopped. Billing paused."
# start-for-demo.sh
echo "Starting backend..."
aws ecs update-service --cluster dis-v7-cluster --service dis-v7-backend --desired-count 1 --region ap-south-1 > /dev/null

echo "Waiting 2 minutes for backend to be healthy..."
sleep 120

echo "Starting frontend..."
aws ecs update-service --cluster dis-v7-cluster --service dis-v7-frontend --desired-count 1 --region ap-south-1 > /dev/null

echo "Waiting 60 seconds..."
sleep 60

echo "Getting your live URLs..."
MSYS_NO_PATHCONV=1 aws logs tail /ecs/dis-v7/backend --since 3m --region ap-south-1 | grep trycloudflare
MSYS_NO_PATHCONV=1 aws logs tail /ecs/dis-v7/frontend --since 3m --region ap-south-1 | grep trycloudflare
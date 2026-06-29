#!/bin/bash
# get-urls.sh — prints live URLs for all services
REGION="ap-south-1"
CLUSTER="dis-v7-cluster"

TASK_ARN=$(aws ecs list-tasks \
  --cluster $CLUSTER \
  --query "taskArns[0]" \
  --output text --region $REGION)

if [ "$TASK_ARN" == "None" ] || [ -z "$TASK_ARN" ]; then
  echo "No running tasks yet. Wait a minute and retry."
  exit 1
fi

ENI_ID=$(aws ecs describe-tasks \
  --cluster $CLUSTER \
  --tasks $TASK_ARN \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
  --output text --region $REGION)

PUBLIC_IP=$(aws ec2 describe-network-interfaces \
  --network-interface-ids $ENI_ID \
  --query "NetworkInterfaces[0].Association.PublicIp" \
  --output text --region $REGION)

echo ""
echo "🚀 Live service URLs:"
echo "  Frontend  → http://$PUBLIC_IP:5000"
echo "  API       → http://$PUBLIC_IP:8000"
echo "  API Docs  → http://$PUBLIC_IP:8000/docs"
echo "  MLflow    → http://$PUBLIC_IP:5001"
echo "  Grafana   → http://$PUBLIC_IP:3000  (admin/admin)"
echo "  Prometheus→ http://$PUBLIC_IP:9090"
echo ""
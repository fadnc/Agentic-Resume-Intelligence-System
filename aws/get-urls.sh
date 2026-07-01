#!/bin/bash
# get-urls-all.sh — prints live URLs for all 4 separate services
REGION="ap-south-1"
CLUSTER="dis-v7-cluster"

get_ip() {
  local service=$1
  TASK_ARN=$(aws ecs list-tasks --cluster $CLUSTER --service-name "dis-v7-$service" \
    --query "taskArns[0]" --output text --region $REGION)

  if [ "$TASK_ARN" == "None" ] || [ -z "$TASK_ARN" ]; then
    echo "no running task"
    return
  fi

  ENI_ID=$(aws ecs describe-tasks --cluster $CLUSTER --tasks $TASK_ARN \
    --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
    --output text --region $REGION)

  aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID \
    --query "NetworkInterfaces[0].Association.PublicIp" \
    --output text --region $REGION
}

echo ""
echo " Live service URLs:"
BACKEND_IP=$(get_ip backend)
echo "  Backend API → http://$BACKEND_IP:8000"
echo "  API Docs    → http://$BACKEND_IP:8000/docs"
echo "  Frontend    → http://$(get_ip frontend):5000"
echo "  MLflow      → http://$(get_ip mlflow):5001"
echo "  Grafana     → http://$(get_ip grafana):3000  (admin/admin)"
echo ""
echo "  Note: backend IP changes on every redeploy."
echo "  Update frontend.json's BACKEND_URL and redeploy frontend"
echo "  whenever the backend IP changes, or set up an ALB for a stable URL."
#!/usr/bin/env bash
# Print public URLs for the four ECS services once their tasks have ENIs/IPs.
set -o pipefail

REGION="ap-south-1"
CLUSTER="dis-v7-cluster"

get_ip() {
  local service="$1"
  local task_arn eni_id public_ip

  if ! task_arn=$(aws ecs list-tasks \
    --cluster "$CLUSTER" \
    --service-name "dis-v7-$service" \
    --desired-status RUNNING \
    --query 'taskArns[0]' --output text --region "$REGION"); then
    echo "AWS_ERROR"
    return 0
  fi

  if [[ -z "$task_arn" || "$task_arn" == "None" ]]; then
    echo "NOT_RUNNING"
    return 0
  fi

  if ! eni_id=$(aws ecs describe-tasks \
    --cluster "$CLUSTER" --tasks "$task_arn" \
    --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value | [0]" \
    --output text --region "$REGION"); then
    echo "AWS_ERROR"
    return 0
  fi

  if [[ -z "$eni_id" || "$eni_id" == "None" ]]; then
    echo "PENDING"
    return 0
  fi

  if ! public_ip=$(aws ec2 describe-network-interfaces \
    --network-interface-ids "$eni_id" \
    --query 'NetworkInterfaces[0].Association.PublicIp' \
    --output text --region "$REGION"); then
    echo "AWS_ERROR"
    return 0
  fi

  if [[ -z "$public_ip" || "$public_ip" == "None" ]]; then
    echo "PENDING"
  else
    echo "$public_ip"
  fi
}

print_service() {
  local label="$1" service="$2" port="$3" ip
  ip=$(get_ip "$service")
  case "$ip" in
    AWS_ERROR) printf '  %-12s AWS CLI request failed; see the error above.\n' "$label" ;;
    NOT_RUNNING) printf '  %-12s no running task (start the ECS service first)\n' "$label" ;;
    PENDING) printf '  %-12s task has no public IP yet; wait a minute and retry\n' "$label" ;;
    *) printf '  %-12s http://%s:%s\n' "$label" "$ip" "$port" ;;
  esac
}

echo
echo " Live service URLs:"
print_service "Backend API" backend 8000
print_service "API Docs" backend 8000/docs
print_service "Frontend" frontend 5000
print_service "MLflow" mlflow 5001
print_service "Grafana" grafana 3000
echo
echo "  Backend IP changes on redeploy. Update frontend.json's BACKEND_URL"
echo "  and redeploy the frontend when it changes, or use an ALB for a stable URL."

#!/bin/bash
# deploy-all-services.sh
# Registers all 4 task definitions and creates/updates their ECS services.
# Run from project root: bash aws/deploy-all-services.sh
set -e

REGION="ap-south-1"
CLUSTER="dis-v7-cluster"
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region $REGION)
SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[*].SubnetId" --output text --region $REGION | tr '\t' ',')
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=dis-v7-sg" --query "SecurityGroups[0].GroupId" --output text --region $REGION)

deploy_service() {
  local name=$1
  local file=$2

  echo "==> Registering task definition: $name"
  aws ecs register-task-definition \
    --cli-input-json file://aws/task-definitions/$file \
    --region $REGION

  echo "==> Creating/updating service: dis-v7-$name"
  aws ecs create-service \
    --cluster $CLUSTER \
    --service-name "dis-v7-$name" \
    --task-definition "dis-v7-$name" \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --region $REGION 2>/dev/null || \
  aws ecs update-service \
    --cluster $CLUSTER \
    --service "dis-v7-$name" \
    --task-definition "dis-v7-$name" \
    --force-new-deployment \
    --region $REGION

  echo "==> $name done"
  echo ""
}

deploy_service "backend"  "backend.json"
deploy_service "frontend" "frontend.json"
deploy_service "mlflow"   "mlflow.json"
deploy_service "grafana"  "grafana.json"

echo " All 4 services deployed. Run get-urls.sh in ~2 min to fetch live IPs."
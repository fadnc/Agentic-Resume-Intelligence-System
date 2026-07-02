#!/bin/bash
# setup-aws.sh — run once to create all AWS infrastructure
# Usage: bash setup-aws.sh YOUR_ACCOUNT_ID
set -e

ACCOUNT_ID=${1:?Usage: bash setup-aws.sh YOUR_ACCOUNT_ID}
REGION="ap-south-1"
CLUSTER="dis-v7-cluster"
SERVICE="dis-v7-service"

echo "==> [1/7] Creating CloudWatch log groups..."
for svc in backend frontend mlflow grafana; do
  aws logs create-log-group \
    --log-group-name "/ecs/dis-v7/$svc" \
    --region $REGION 2>/dev/null || echo "  /ecs/dis-v7/$svc already exists"
done

echo "==> [2/7] Storing GROQ_API_KEY in Secrets Manager..."
read -s -p "  Enter your GROQ_API_KEY: " GROQ_KEY; echo
aws secretsmanager create-secret \
  --name "dis-v7/groq-api-key" \
  --secret-string "$GROQ_KEY" \
  --region $REGION 2>/dev/null || \
aws secretsmanager put-secret-value \
  --secret-id "dis-v7/groq-api-key" \
  --secret-string "$GROQ_KEY" \
  --region $REGION
echo "  Secret stored."

echo "==> [3/7] Creating ECS task execution role..."
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{
      "Effect":"Allow",
      "Principal":{"Service":"ecs-tasks.amazonaws.com"},
      "Action":"sts:AssumeRole"
    }]
  }' 2>/dev/null || echo "  Role already exists"

aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Allow reading Secrets Manager
aws iam put-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-name AllowSecretsManager \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[{
      \"Effect\":\"Allow\",
      \"Action\":[\"secretsmanager:GetSecretValue\"],
      \"Resource\":\"arn:aws:secretsmanager:$REGION:$ACCOUNT_ID:secret:dis-v7/*\"
    }]
  }"
echo "  Role ready."

echo "==> [4/7] Patching task definition with your account ID..."
sed -i "s/YOUR_ACCOUNT_ID/$ACCOUNT_ID/g" ecs-task-definition.json
echo "  Patched."

echo "==> [5/7] Registering ECS task definition..."
aws ecs register-task-definition \
  --cli-input-json file://ecs-task-definition.json \
  --region $REGION
echo "  Task definition registered."

echo "==> [6/7] Creating ECS cluster..."
aws ecs create-cluster \
  --cluster-name $CLUSTER \
  --capacity-providers FARGATE \
  --region $REGION 2>/dev/null || echo "  Cluster already exists"

echo "==> [7/7] Creating ECS service..."
# Get default VPC and subnets
VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text --region $REGION)

SUBNETS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "Subnets[*].SubnetId" --output text --region $REGION | tr '\t' ',')

# Create security group
SG_ID=$(aws ec2 create-security-group \
  --group-name dis-v7-sg \
  --description "dis-v7 ECS security group" \
  --vpc-id $VPC_ID \
  --region $REGION \
  --query "GroupId" --output text 2>/dev/null || \
aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=dis-v7-sg" \
  --query "SecurityGroups[0].GroupId" --output text --region $REGION)

# Open ports
for PORT in 8000 5000 5001 3000; do
  aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp --port $PORT --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true
done

aws ecs create-service \
  --cluster $CLUSTER \
  --service-name $SERVICE \
  --task-definition dis-v7 \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[$SUBNETS],
    securityGroups=[$SG_ID],
    assignPublicIp=ENABLED
  }" \
  --region $REGION 2>/dev/null || echo "  Service already exists"

echo ""
echo "AWS infrastructure ready!"
echo ""
echo "Get your public IP once the task starts (~2 min):"
echo "  bash get-urls.sh"
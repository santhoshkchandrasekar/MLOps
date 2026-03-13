# Terraform Lab - Custom Configuration
# Region: us-west-2 (Oregon) - different from professor's us-east-1
# Changes: t3.micro instance, S3 bucket, detailed tags

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

# EC2 Instance - using t3.micro + Amazon Linux 2023 AMI for us-west-2
resource "aws_instance" "myec2" {
  ami           = "ami-05572e392e80aee89"
  instance_type = "t3.micro"

  tags = {
    Name        = "Santhosh-EC2"
    Environment = "Dev"
    Project     = "MLOps-TerraformLab"
    Owner       = "Santhosh"
  }
}

# S3 Bucket - not in professor's version
resource "aws_s3_bucket" "mylabbucket" {
  bucket = "santhosh-terraform-lab-bucket-2026"

  tags = {
    Name        = "Santhosh-Lab-Bucket"
    Environment = "Dev"
    Project     = "MLOps-TerraformLab"
    Owner       = "Santhosh"
  }
}

# VPC
resource "aws_vpc" "myvpc" {
  cidr_block = "10.0.0.0/16"

  tags = {
    Name        = "Santhosh-VPC"
    Environment = "Dev"
    Project     = "MLOps-TerraformLab"
    Owner       = "Santhosh"
  }
}

# Subnet
resource "aws_subnet" "mysubnet1" {
  vpc_id     = aws_vpc.myvpc.id
  cidr_block = "10.0.1.0/24"

  tags = {
    Name        = "Santhosh-Subnet1"
    Environment = "Dev"
    Project     = "MLOps-TerraformLab"
    Owner       = "Santhosh"
  }
}
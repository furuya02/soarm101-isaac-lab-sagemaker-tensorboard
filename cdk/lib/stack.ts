import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

export interface SoarmStackProps extends cdk.StackProps {
  projectName: string;
  bucketSuffix: string;
}

export class SoarmStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: SoarmStackProps) {
    super(scope, id, props);
    const { projectName, bucketSuffix } = props;

    const bucket = new s3.Bucket(this, "ArtifactBucket", {
      bucketName: `${projectName}-${bucketSuffix}`,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      lifecycleRules: [{ expiration: cdk.Duration.days(30) }],
    });

    const repository = new ecr.Repository(this, "TrainingImageRepo", {
      repositoryName: projectName,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      imageScanOnPush: true,
      lifecycleRules: [
        {
          maxImageAge: cdk.Duration.days(1),
          tagStatus: ecr.TagStatus.UNTAGGED,
        },
      ],
    });

    const sagemakerRole = new iam.Role(this, "SageMakerExecutionRole", {
      roleName: `${projectName}-sagemaker-execution-role`,
      assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess"),
      ],
    });
    bucket.grantReadWrite(sagemakerRole);
    repository.grantPull(sagemakerRole);

    new cdk.CfnOutput(this, "BucketName", { value: bucket.bucketName });
    new cdk.CfnOutput(this, "EcrRepositoryUri", {
      value: repository.repositoryUri,
    });
    new cdk.CfnOutput(this, "SageMakerRoleArn", {
      value: sagemakerRole.roleArn,
    });
  }
}

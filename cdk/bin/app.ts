#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { SoarmStack } from "../lib/stack";

const app = new cdk.App();
const accountId = app.node.tryGetContext("account_id") ?? process.env.CDK_DEFAULT_ACCOUNT;
if (!accountId) throw new Error("account_id context or CDK_DEFAULT_ACCOUNT required");
const region = app.node.tryGetContext("region") ?? process.env.CDK_DEFAULT_REGION ?? "ap-northeast-1";

new SoarmStack(app, "SoarmStack", {
  env: { account: accountId, region },
  projectName: "soarm101-isaac-lab-sagemaker-tensorboard",
  bucketSuffix: app.node.tryGetContext("bucket_suffix") ?? accountId,
});

# smworkshop

Most examples are adapted from the [SageMaker example](https://github.com/awslabs/amazon-sagemaker-examples) repository.

 ID | Levl | Data | Problem | Built-in Algo | ML Framework | Custom Algo | Custom Infer | Custom Container | Extra lib | Param tuning | Model monitor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | 0 | Churn|Cls |AutoPilot | | | | | | | |
| 02 | 1 | BH*|Reg |XGB | | | | | | |Yes |
| 03 | 1 | BH*|Reg | |SKLearn |RF-MLP |Input | | |RF params | |
| 04 | 2 | BH*|Reg | |PyTorch |MLP |Input | |Plotting |NN-arch | |
| 05 | 3 | Census|Cls || FastAI|Tabular |Pred | | | | |
| 06 | 4 | IRIS|Cls | |SKLearn | | |Ubuntu:16.04 | | | |
\* Boston Housing
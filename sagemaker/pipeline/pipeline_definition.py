"""
SageMaker Pipeline definition:
  Preprocess → Train → Evaluate → Condition → Register + Deploy.

Creates an 'edtriage-train-pipeline' with parameterised inputs so the team
can swap instance types and architectures without editing this file.

The pipeline automatically compares each trained model against the current
champion (stored in S3).  If the new model's macro-F1 is strictly better,
it is registered in the Model Registry and deployed to a live endpoint.

The TrainingStep reads from TrainDataUri / ValidationDataUri parameters so it
can be run independently of the ProcessingStep (e.g. when splits already exist
in S3).  When run_pipeline.py detects existing splits it passes those URIs
directly and omits the ProcessingStep.
"""

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

PIPELINE_NAME = "edtriage-train-pipeline"

# Pre-built framework container images (us-east-1)
SKLEARN_FRAMEWORK_VERSION = "1.2-1"
PYTORCH_FRAMEWORK_VERSION = "2.1.0"
PYTORCH_PY_VERSION        = "py310"

DEFAULT_SPLITS_PREFIX = "Data_Output/splits"


def get_pipeline(
    role: str,
    epochs: int,
    training_instance_type_str: str,
    architecture: str,
    region: str = "us-east-1",
    default_bucket: str | None = None,
    pipeline_name: str = PIPELINE_NAME,
    skip_preprocessing: bool = False,
    endpoint_instance_type_str: str = "ml.m5.xlarge",
) -> Pipeline:
    """Build and return the SageMaker Pipeline object (does not execute it).

    Args:
        architecture: Model architecture to train (e.g. "mock", "arch4").
            Determines which model module is loaded by the training dispatcher.
        skip_preprocessing: When True, omit the ProcessingStep and wire
            TrainingStep directly to the TrainDataUri / ValidationDataUri
            pipeline parameters.
        endpoint_instance_type_str: Instance type for the inference endpoint.
    """

    session = sagemaker.Session(
        default_bucket=default_bucket,
    )
    if default_bucket is None:
        default_bucket = session.default_bucket()

    splits_base = f"s3://{default_bucket}/{DEFAULT_SPLITS_PREFIX}"

    model_package_group_name = f"edtriage-{architecture}"

    # ── Pipeline Parameters ───────────────────────────────────────────────────
    input_data_uri = ParameterString(
        name="InputDataUri",
        default_value="s3://ed-triage-capstone-group7/Data_Output/consolidated_dataset_features.csv",
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.t3.medium",
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value=training_instance_type_str,
    )
    training_instance_count = ParameterString(
        name="TrainingInstanceCount",
        default_value="1",
    )
    train_data_uri = ParameterString(
        name="TrainDataUri",
        default_value=f"{splits_base}/train/",
    )
    validation_data_uri = ParameterString(
        name="ValidationDataUri",
        default_value=f"{splits_base}/validation/",
    )
    test_data_uri = ParameterString(
        name="TestDataUri",
        default_value=f"{splits_base}/test/",
    )
    endpoint_instance_type = ParameterString(
        name="EndpointInstanceType",
        default_value=endpoint_instance_type_str,
    )
    endpoint_name = ParameterString(
        name="EndpointName",
        default_value="edtriage-live",
    )

    # ── Step 2: Training ──────────────────────────────────────────────────────
    estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve(
            framework="pytorch",
            region=region,
            version=PYTORCH_FRAMEWORK_VERSION,
            py_version=PYTORCH_PY_VERSION,
            instance_type=training_instance_type_str,
            image_scope="training",
        ),
        role=role,
        instance_count=int(training_instance_count.default_value),
        instance_type=training_instance_type,
        entry_point="steps/train.py",
        source_dir="sagemaker",
        sagemaker_session=session,
        output_path=f"s3://{default_bucket}/models",
        hyperparameters={
            "epochs": str(epochs),
            "architecture": architecture,
        },
    )

    pipeline_parameters = [
        training_instance_type,
        training_instance_count,
        train_data_uri,
        validation_data_uri,
    ]

    if skip_preprocessing:
        # ── Training-only pipeline ────────────────────────────────────────────
        training_step = TrainingStep(
            name="Train",
            estimator=estimator,
            inputs={
                "train": TrainingInput(
                    s3_data=train_data_uri,
                    content_type="text/csv",
                ),
                "validation": TrainingInput(
                    s3_data=validation_data_uri,
                    content_type="text/csv",
                ),
            },
        )
        test_data_source = test_data_uri
        steps = [training_step]

    else:
        # ── Full pipeline: Preprocess → Train ─────────────────────────────────
        sklearn_processor = SKLearnProcessor(
            framework_version=SKLEARN_FRAMEWORK_VERSION,
            role=role,
            instance_type=processing_instance_type,
            instance_count=1,
            sagemaker_session=session,
        )

        processing_step = ProcessingStep(
            name="Preprocess",
            processor=sklearn_processor,
            code="sagemaker/steps/preprocess.py",
            inputs=[
                ProcessingInput(
                    source=input_data_uri,
                    destination="/opt/ml/processing/input/features",
                    input_name="features",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/output/train",
                    destination=f"{splits_base}/train",
                ),
                ProcessingOutput(
                    output_name="validation",
                    source="/opt/ml/processing/output/validation",
                    destination=f"{splits_base}/validation",
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/output/test",
                    destination=f"{splits_base}/test",
                ),
            ],
        )

        training_step = TrainingStep(
            name="Train",
            estimator=estimator,
            inputs={
                "train": TrainingInput(
                    s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                        "train"
                    ].S3Output.S3Uri,
                    content_type="text/csv",
                ),
                "validation": TrainingInput(
                    s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                        "validation"
                    ].S3Output.S3Uri,
                    content_type="text/csv",
                ),
            },
        )

        test_data_source = processing_step.properties.ProcessingOutputConfig.Outputs[
            "test"
        ].S3Output.S3Uri
        pipeline_parameters += [input_data_uri, processing_instance_type]
        steps = [processing_step, training_step]

    # ── Step 3: Evaluate (compare against champion) ──────────────────────────
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluate_processor = SKLearnProcessor(
        framework_version=SKLEARN_FRAMEWORK_VERSION,
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        sagemaker_session=session,
        env={"BUCKET_NAME": default_bucket},
    )

    evaluate_step = ProcessingStep(
        name="Evaluate",
        processor=evaluate_processor,
        code="sagemaker/steps/evaluate.py",
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
                input_name="model",
            ),
            ProcessingInput(
                source=test_data_source,
                destination="/opt/ml/processing/input/test",
                input_name="test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output/evaluation",
                destination=f"s3://{default_bucket}/evaluation",
            ),
        ],
        property_files=[evaluation_report],
    )
    steps.append(evaluate_step)

    # ── Step 4: Condition (is this the new champion?) ────────────────────────
    is_champion_condition = ConditionEquals(
        left=JsonGet(
            step_name=evaluate_step.name,
            property_file=evaluation_report,
            json_path="is_new_champion",
        ),
        right="true",
    )

    # ── Step 5a: RegisterModel (only if new champion) ────────────────────────
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge", "ml.g5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status="Approved",
    )

    # ── Step 5b: Deploy (only if new champion) ───────────────────────────────
    inference_image_uri = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        version=PYTORCH_FRAMEWORK_VERSION,
        py_version=PYTORCH_PY_VERSION,
        instance_type=endpoint_instance_type_str,
        image_scope="inference",
    )

    deploy_processor = SKLearnProcessor(
        framework_version=SKLEARN_FRAMEWORK_VERSION,
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        sagemaker_session=session,
        env={
            "BUCKET_NAME": default_bucket,
            "ENDPOINT_NAME": endpoint_name.default_value,
            "ENDPOINT_INSTANCE_TYPE": endpoint_instance_type_str,
            "ROLE_ARN": role,
            "CONTAINER_IMAGE": inference_image_uri,
            "ARCHITECTURE": architecture,
        },
    )

    deploy_step = ProcessingStep(
        name="Deploy",
        processor=deploy_processor,
        code="sagemaker/steps/deploy.py",
        inputs=[],
        outputs=[],
    )
    deploy_step.add_depends_on(register_step.steps)

    condition_step = ConditionStep(
        name="CheckChampion",
        conditions=[is_champion_condition],
        if_steps=[register_step, deploy_step],
        else_steps=[],
    )
    steps.append(condition_step)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline_parameters += [test_data_uri, endpoint_instance_type, endpoint_name]

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=pipeline_parameters,
        steps=steps,
        sagemaker_session=session,
    )

    return pipeline

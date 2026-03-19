"""
SageMaker Pipeline definition: ProcessingStep → TrainingStep.

Creates an 'edtriage-train-pipeline' with parameterised inputs so the team
can swap instance types and training scripts without editing this file.

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
    region: str = "us-east-1",
    default_bucket: str | None = None,
    pipeline_name: str = PIPELINE_NAME,
    training_script: str = "train_mock.py",
    skip_preprocessing: bool = False,
) -> Pipeline:
    """Build and return the SageMaker Pipeline object (does not execute it).

    Args:
        skip_preprocessing: When True, omit the ProcessingStep and wire
            TrainingStep directly to the TrainDataUri / ValidationDataUri
            pipeline parameters.
    """

    session = sagemaker.Session(
        default_bucket=default_bucket,
    )
    if default_bucket is None:
        default_bucket = session.default_bucket()

    splits_base = f"s3://{default_bucket}/{DEFAULT_SPLITS_PREFIX}"

    # Derive model package group from training script: train_arch4.py → edtriage-arch4
    arch_name = training_script.replace("train_", "").replace(".py", "")
    model_package_group_name = f"edtriage-{arch_name}"

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
        default_value=f"{splits_base}/train",
    )
    validation_data_uri = ParameterString(
        name="ValidationDataUri",
        default_value=f"{splits_base}/validation",
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
        entry_point=training_script,
        source_dir="scripts",
        sagemaker_session=session,
        output_path=f"s3://{default_bucket}/models",
        hyperparameters={
            "epochs": str(epochs),
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
            code="scripts/preprocess.py",
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

        pipeline_parameters += [input_data_uri, processing_instance_type]
        steps = [processing_step, training_step]

    # ── RegisterModel step (always runs after Train) ──────────────────────────
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge", "ml.g5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval",
    )
    steps.append(register_step)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=pipeline_parameters,
        steps=steps,
        sagemaker_session=session,
    )

    return pipeline

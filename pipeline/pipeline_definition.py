"""
SageMaker Pipeline definition: ProcessingStep → TrainingStep.

Creates an 'edtriage-train-pipeline' with parameterised inputs so the team
can swap instance types and training scripts without editing this file.
"""

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline

PIPELINE_NAME = "edtriage-train-pipeline"

# Pre-built framework container images (us-east-1)
SKLEARN_FRAMEWORK_VERSION = "1.2-1"
PYTORCH_FRAMEWORK_VERSION = "2.1.0"
PYTORCH_PY_VERSION        = "py310"


def get_pipeline(
    role: str,
    region: str = "us-east-1",
    default_bucket: str | None = None,
    pipeline_name: str = PIPELINE_NAME,
) -> Pipeline:
    """Build and return the SageMaker Pipeline object (does not execute it)."""

    session = sagemaker.Session(
        default_bucket=default_bucket,
    )
    if default_bucket is None:
        default_bucket = session.default_bucket()

    # ── Pipeline Parameters ───────────────────────────────────────────────────
    input_data_uri = ParameterString(
        name="InputDataUri",
        default_value="s3://ed-triage-capstone-group7/Data_Output/consolidated_dataset_features.csv",
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.xlarge",
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.xlarge",
    )
    training_instance_count = ParameterString(
        name="TrainingInstanceCount",
        default_value="1",
    )
    training_script = ParameterString(
        name="TrainingScript",
        default_value="train_mock.py",
    )

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
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
                destination=f"s3://{default_bucket}/Data_Output/splits/train",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{default_bucket}/Data_Output/splits/validation",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{default_bucket}/Data_Output/splits/test",
            ),
        ],
    )

    # ── Step 2: Training ──────────────────────────────────────────────────────
    estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve(
            framework="pytorch",
            region=region,
            version=PYTORCH_FRAMEWORK_VERSION,
            py_version=PYTORCH_PY_VERSION,
            instance_type="ml.m5.xlarge",  # used only for image lookup
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
            "epochs": "1",
        },
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

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data_uri,
            processing_instance_type,
            training_instance_type,
            training_instance_count,
            training_script,
        ],
        steps=[processing_step, training_step],
        sagemaker_session=session,
    )

    return pipeline

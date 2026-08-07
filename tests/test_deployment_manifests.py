"""
Tests for deployment/manifests.py.

Reads this repo's real kubernetes/*.yaml files (no mocking needed —
these are static, committed, read-only inputs, same pattern as
deployment_pipeline_status.py's "reflects this real repository" test)
and asserts the built bundle is correct. No kubectl call happens here.
"""

import yaml

from deployment.manifests import (
    DEPLOYMENT_MANIFEST,
    HPA_MANIFEST,
    NAMESPACE_MANIFEST,
    SERVICE_MANIFEST,
    build_manifest_bundle,
)


def test_build_manifest_bundle_swaps_only_the_container_image():
    manifest_yaml, deployment_name, namespace = build_manifest_bundle(
        "ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:abc123"
    )

    documents = list(yaml.safe_load_all(manifest_yaml))
    deployment_doc = next(doc for doc in documents if doc["kind"] == "Deployment")

    assert (
        deployment_doc["spec"]["template"]["spec"]["containers"][0]["image"]
        == "ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:abc123"
    )
    # Everything else in the deployment comes through unmodified from
    # the committed manifest.
    assert deployment_doc["spec"]["replicas"] == 2
    assert (
        deployment_doc["spec"]["template"]["spec"]["containers"][0]["ports"][0][
            "containerPort"
        ]
        == 8000
    )


def test_build_manifest_bundle_reads_name_and_namespace_from_the_real_manifest():
    _, deployment_name, namespace = build_manifest_bundle("some-image:tag")

    real_deployment = yaml.safe_load(DEPLOYMENT_MANIFEST.read_text())
    assert deployment_name == real_deployment["metadata"]["name"]
    assert namespace == real_deployment["metadata"]["namespace"]


def test_build_manifest_bundle_includes_namespace_service_and_hpa_verbatim():
    manifest_yaml, _, _ = build_manifest_bundle("some-image:tag")
    documents = list(yaml.safe_load_all(manifest_yaml))
    kinds = {doc["kind"] for doc in documents}

    assert kinds == {"Namespace", "Deployment", "Service", "HorizontalPodAutoscaler"}

    namespace_doc = next(doc for doc in documents if doc["kind"] == "Namespace")
    service_doc = next(doc for doc in documents if doc["kind"] == "Service")
    hpa_doc = next(doc for doc in documents if doc["kind"] == "HorizontalPodAutoscaler")

    assert namespace_doc == yaml.safe_load(NAMESPACE_MANIFEST.read_text())
    assert service_doc == yaml.safe_load(SERVICE_MANIFEST.read_text())
    assert hpa_doc == yaml.safe_load(HPA_MANIFEST.read_text())


def test_build_manifest_bundle_is_valid_multi_document_yaml():
    manifest_yaml, _, _ = build_manifest_bundle("some-image:tag")

    documents = list(yaml.safe_load_all(manifest_yaml))
    assert len(documents) == 4

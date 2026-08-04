import numpy as np
import pytest

from msc_cali.geometry import solve_rigid_transform


def test_solve_rigid_transform_recovers_known_transform() -> None:
    camera = np.asarray(
        [
            [-0.3, -0.2, 0.7],
            [0.2, -0.1, 0.9],
            [-0.1, 0.3, 1.1],
            [0.3, 0.2, 1.3],
            [-0.25, 0.1, 1.4],
            [0.1, -0.3, 1.2],
            [0.0, 0.0, 0.8],
            [0.25, -0.25, 1.5],
        ]
    )
    angle = np.deg2rad(35.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.asarray([0.45, -0.12, 0.31])
    robot = camera @ rotation.T + translation

    result = solve_rigid_transform(camera, robot)

    np.testing.assert_allclose(result.transform.rotation, rotation, atol=1e-10)
    np.testing.assert_allclose(result.transform.translation_m, translation, atol=1e-10)
    assert result.rmse_m < 1e-10
    np.testing.assert_allclose(result.transform.inverse().apply(robot), camera, atol=1e-10)


def test_solve_rejects_planar_geometry() -> None:
    camera = np.asarray([[x, y, 1.0] for x in (-0.2, 0.0, 0.2) for y in (-0.2, 0.2)])
    with pytest.raises(ValueError, match="planar or collinear"):
        solve_rigid_transform(camera, camera.copy())


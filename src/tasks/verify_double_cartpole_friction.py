from pathlib import Path

import mujoco


def main() -> None:
    root = Path(".")
    model = mujoco.MjModel.from_xml_path(
        str(root / "models" / "double_cart_pole" / "scene.xml")
    )
    joint = model.joint("hinge_2")
    dof = joint.dofadr
    damping = float(model.dof_damping[dof])
    frictionloss = float(model.dof_frictionloss[dof])
    print(f"hinge_2 damping: {damping}")
    print(f"hinge_2 frictionloss: {frictionloss}")


if __name__ == "__main__":
    main()

import os
import numpy as np
import pydot
import matplotlib.pyplot as plt
from IPython.display import SVG, display

from pydrake.common import temp_directory
from pydrake.geometry import StartMeshcat, QueryObject
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.visualization import AddDefaultVisualization
from pydrake.common.value import Value

from controllers.PID import PD_gravity

meshcat = StartMeshcat()
initPos = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

def get_relative_path(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, path))

def create_scene(sim_time_step):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    parser = Parser(plant)

    # Load the robot
    panda_path = get_relative_path("../../../models/descriptions/robots/panda_fr3/urdf/panda_fr3_nohand.urdf")
    parser.AddModelsFromUrl("file://" + panda_path)

    # Load fixed and movable boxes from hybrids folder
    fixed_box_path = get_relative_path("../../../models/descriptions/hybrids/fixed_box.sdf")
    movable_box_path = get_relative_path("../../../models/descriptions/hybrids/movable_box.sdf")

    parser.AddModelsFromUrl("file://" + fixed_box_path)
    parser.AddModelsFromUrl("file://" + movable_box_path)

    # Weld panda base to world
    X_panda_base = RigidTransform(p=[0.0, 0.0, 0.0])
    plant.WeldFrames(plant.GetFrameByName("world"), plant.GetFrameByName("panda_link0"), X_panda_base)

    # Optionally weld fixed box if not static
    try:
        plant.WeldFrames(
            plant.GetFrameByName("world"),
            plant.GetFrameByName("box_link", plant.GetModelInstanceByName("fixed_box"))
        )
    except:
        pass

    plant.Finalize()
    print("Contact model: ", plant.get_contact_model())
    print("Is discrete (time_step > 0)?", plant.time_step() > 0)

    panda_id = plant.GetModelInstanceByName("panda")
    plant.SetDefaultPositions(panda_id, initPos)

    Kp = 15 * np.array([120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0])
    Kd = 10 * np.array([8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 2.0])

    controller = builder.AddNamedSystem("PD+G controller", PD_gravity(plant, Kp, Kd, csv_filename='actuation_data.csv'))
    desired_q = [0.0, -0.285, 0.0, -2.356, 0.0, 1.571, 0.785]
    des_pos = builder.AddNamedSystem("Desired Position", ConstantVectorSource(desired_q))

    builder.Connect(plant.get_state_output_port(panda_id), controller.GetInputPort("Current_state"))
    builder.Connect(des_pos.get_output_port(), controller.GetInputPort("Desired_state"))
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port(panda_id))

    # Optional dummy connection
    builder.Connect(plant.GetOutputPort("panda_net_actuation"), controller.GetInputPort("actuation"))

    AddDefaultVisualization(builder=builder, meshcat=meshcat)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    return diagram, diagram_context, plant

# Run simulation
diagram, diagram_context, plant = create_scene(0.00001)
simulator = Simulator(diagram)
simulator.Initialize()
simulator.set_target_realtime_rate(1)

meshcat.StartRecording()
simulator.AdvanceTo(15.0)
meshcat.PublishRecording()

svg_data = diagram.GetGraphvizString(max_depth=2)
graph = pydot.graph_from_dot_data(svg_data)[0]
graph.write_png("block_diagram_panda_boxes.png")
print("Block diagram saved as: block_diagram_panda_boxes.png")

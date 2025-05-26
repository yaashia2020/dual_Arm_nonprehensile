
import os
import numpy as np
import pydot
from pydrake.geometry import StartMeshcat
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.visualization import AddDefaultVisualization
from controllers.PID import PD_gravity

meshcat = StartMeshcat()
initPos = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.0]

def get_relative_path(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, path))

def create_scene(sim_time_step):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    parser = Parser(plant)

    # Use duplicated URDFs with different <robot name="...">
    #panda_path_1 = get_relative_path("../../../models/descriptions/robots/panda_fr3/urdf/panda_fr3_nohand.urdf")
    #panda_path_2 = get_relative_path("../../../models/descriptions/robots/panda_fr3/urdf/panda_fr3_nohand2.urdf")
    panda_path_1 = "/home/art/dual_arm_example/models/robots/panda_fr3/urdf/panda_fr3_nohand.urdf"
    panda_path_2 = "/home/art/dual_arm_example/models/robots/panda_fr3/urdf/panda_fr3_nohand2.urdf"

    parser.AddModelsFromUrl("file://" + panda_path_1)  # loads 'panda'
    parser.AddModelsFromUrl("file://" + panda_path_2)  # loads 'panda_1'

    fixed_box_path = get_relative_path("/home/art/dual_arm_example/models/boxes/fixed_box.sdf")
    movable_box_path = get_relative_path("/home/art/dual_arm_example/models/boxes/movable_box.sdf")
    parser.AddModelsFromUrl("file://" + fixed_box_path)
    parser.AddModelsFromUrl("file://" + movable_box_path)

    # Weld both Pandas
    plant.WeldFrames(plant.GetFrameByName("world"), plant.GetFrameByName("panda_link0", plant.GetModelInstanceByName("panda")), RigidTransform(p=[0.0, 0.0, 0.0]))
    plant.WeldFrames(plant.GetFrameByName("world"), plant.GetFrameByName("panda_link0", plant.GetModelInstanceByName("panda_1")), RigidTransform(RollPitchYaw(0, 0, np.pi), [1.4, 0.0, 0.0]))

    #plant.WeldFrames(plant.GetFrameByName("world"), plant.GetFrameByName("box_link", plant.GetModelInstanceByName("fixed_box")))

    plant.Finalize()
    print("Contact model:", plant.get_contact_model())
    print("Is discrete:", plant.time_step() > 0)

    Kp1 = 15 * np.array([120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0])
    Kd1 = 10 * np.array([8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 2.0])

    Kp2 = 15 * np.array([120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0])
    Kp2 = 15 * np.array([120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0])
    Kp2 = 15 * np.array([120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0])
    Kd2 = 10* np.array([8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 2.0])
    desired_q1 = [ 0, 0, 0, -1.58, 1.58*2, 3.14+0.5, 0+1 ]
    desired_q1 = [ 0, 0.3, 0, -1.58, 1.58, 3.14, 0+1 ]
    desired_q2 = [ 0, 0.6, 0, -1.58, 1.58, 3.14, 0-1 ]

    #initPos = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

    initPos1 = [ 0, 0, 0, -1.58*0, 1.58*0, 3.14*0, 0+1 ]
    initPos2 = [ 0, 0, 0, -1.58*0, 1.58*0, 3.14*0, 0-1 ]
    panda1_id = plant.GetModelInstanceByName("panda")
    panda2_id = plant.GetModelInstanceByName("panda_1")

    plant.SetDefaultPositions(panda1_id, initPos1)
    plant.SetDefaultPositions(panda2_id, initPos2)

    # Controller 1
    controller1 = builder.AddNamedSystem("PD+G controller 1", PD_gravity(plant, Kp1, Kd1))
    des_pos1 = builder.AddNamedSystem("Desired Position 1", ConstantVectorSource(desired_q1))
    builder.Connect(plant.get_state_output_port(panda1_id), controller1.GetInputPort("Current_state"))
    builder.Connect(des_pos1.get_output_port(), controller1.GetInputPort("Desired_state"))
    builder.Connect(controller1.get_output_port(), plant.get_actuation_input_port(panda1_id))
    builder.Connect(plant.GetOutputPort("panda_net_actuation"), controller1.GetInputPort("actuation"))

    # Controller 2
    controller2 = builder.AddNamedSystem("PD+G controller 2", PD_gravity(plant, Kp2, Kd2))
    des_pos2 = builder.AddNamedSystem("Desired Position 2", ConstantVectorSource(desired_q2))
    builder.Connect(plant.get_state_output_port(panda2_id), controller2.GetInputPort("Current_state"))
    builder.Connect(des_pos2.get_output_port(), controller2.GetInputPort("Desired_state"))
    builder.Connect(controller2.get_output_port(), plant.get_actuation_input_port(panda2_id))
    builder.Connect(plant.GetOutputPort("panda_1_net_actuation"), controller2.GetInputPort("actuation"))

    AddDefaultVisualization(builder=builder, meshcat=meshcat)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    return diagram, diagram_context, plant

# Run simulation
diagram, diagram_context, plant = create_scene(0.00001)
simulator = Simulator(diagram)
html_path = "/home/art/meshcat_recording.html"
with open(html_path, "w") as f:
    f.write(meshcat.StaticHtml())
    f.write(meshcat.StaticHtml())
    f.write(meshcat.StaticHtml())
print(f"Recording saved to: {html_path}")
simulator.Initialize()
simulator.set_target_realtime_rate(1)



meshcat.StartRecording()
simulator.AdvanceTo(2.0)
meshcat.PublishRecording()

# Save HTML22
# Save Meshcat recording to HTML
html_path = "/home/art/drake_brubotics-main/meshcat_recording.html"

html_data = meshcat.StaticHtml()
print("Recording size (characters):", len(html_data))

if len(html_data) > 0:
    with open(html_path, "w") as f:
        f.write(html_data)
    print(f"Recording saved to: {html_path}")
else:
    print("Meshcat recording appears to be empty. Did any geometry move?")


svg_data = diagram.GetGraphvizString(max_depth=2)
graph = pydot.graph_from_dot_data(svg_data)[0]
graph.write_png("block_diagram_dual_panda_final_fixed.png")
print("Block diagram saved as: block_diagram_dual_panda_final_fixed.png")
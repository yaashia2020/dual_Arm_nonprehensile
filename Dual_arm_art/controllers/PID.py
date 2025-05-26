import csv
from pydrake.all import *

class PD_gravity(LeafSystem):
    def __init__(self, plant, kp, kd, csv_filename='actuation_data.csv'):
        super().__init__()

        # Declare input ports for desired and current states
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=14)
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=7)

        # Declare an input port with 7 elements (for the 7 joints of the Panda robot)
        self.net_actuation = self.DeclareVectorInputPort("actuation", BasicVector(7))

        # PD+G gains (Kp and Kd)
        self.Kp_ = kp
        self.Kd_ = kd

        # Store plant and context for dynamics calculations
        self.plant, self.plant_context_ad = plant, plant.CreateDefaultContext()
        self.model_index = self.plant.GetModelInstanceByName("panda")

        # Declare discrete state and output port for control input (tau_u)
        state_index = self.DeclareDiscreteState(7)  # 7 state variables.
        self.DeclareStateOutputPort("tau_u", state_index)  # output: y=x.
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/100,  # One millisecond time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.compute_tau_u)  # Call the Update method defined below.
        
        # Create a periodic event to log the actuation data
        self.DeclarePeriodicPublishEvent(0.001, 0, self.PublishEvent) 

        # Initialize CSV file for saving data
        self.csv_filename = csv_filename
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([f"Joint {i+1}" for i in range(7)])  # Header row

    def PublishEvent(self, context):
        # Log the actuation data
        actuation_data = self.net_actuation.Eval(context)
        self.csv_writer.writerow(actuation_data)
        #print(f"Actuation data logged: {actuation_data}")
    
    def compute_tau_u(self, context, discrete_state):
        num_positions = self.plant.num_positions(self.model_index)
        
        # Evaluate the input ports
        self.q_d = self._desired_state_port.Eval(context)
        self.q = self._current_state_port.Eval(context)

        # Compute gravity forces for the current state
        self.plant.SetPositionsAndVelocities(self.plant_context_ad, self.model_index, self.q)
        
        gravity = -self.plant.CalcGravityGeneralizedForces(self.plant_context_ad)    
        
        # Extract relevant components of gravity (for the panda)
        tau_gravity = gravity[:num_positions]

        # Compute the control input (tau)
        tau = self.Kp_ * (self.q_d - self.q[:num_positions]) - self.Kd_ * self.q[num_positions:] + tau_gravity

        # Update the output port = state
        discrete_state.get_mutable_vector().SetFromVector(tau)

    def __del__(self):
        # Close the CSV file when the object is deleted
        if hasattr(self, 'csv_file'):
            self.csv_file.close()

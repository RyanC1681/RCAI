import logging, warnings
import numpy as np
import time
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from pathlib import Path
from ROAR.agent_module.pure_pursuit_agent \
    import PurePursuitAgent
from ROAR_Sim.carla_client.carla_runner import CarlaRunner
from typing import Tuple
from prettytable import PrettyTable



# old def compute_score(carla_runner: CarlaRunner, min_bounding_box = np.array([0,-2,30]), max_bounding_box = np.array([60,2,60])) -> Tuple[float, int, bool]:

from ROAR.agent_module.pid_agent import PIDAgent

from pit_stop import PitStop as PitStop


#def compute_score(carla_runner: CarlaRunner) -> Tuple[float, int, int]:
#def compute_score(carla_runner: CarlaRunner, min_bounding_box=np.array([5, -5, 0]),
#                  max_bounding_box=np.array([13, 5, 50])) -> Tuple[float, int, int]:
def compute_score(carla_runner: CarlaRunner, min_bounding_box=np.array([-815, 20, -760]),
                  max_bounding_box=np.array([-770, 120, -600])) -> Tuple[float, int, int]:
    """
    Calculates the score of the vehicle upon completion of the track based on certain metrics
    Args:
        carla_runner ():
        min_bounding_box ():
        max_bounding_box ():

    Returns:
        time_elapsed:
        num_collision: number of collisions during simulation
        laps_completed: Number of laps completed

    """
    time_elapsed: float = carla_runner.end_simulation_time - carla_runner.start_simulation_time
    num_collision: int = carla_runner.agent_collision_counter
    #laps_completed = 0 if carla_runner.completed_lap_count < 0 else carla_runner.completed_lap_count
    laps_completed = min(0, carla_runner.completed_lap_count)
    return time_elapsed, num_collision, laps_completed

#def run(agent_class, agent_config_file_path: Path, carla_config_file_path: Path, num_laps: int = 10) -> Tuple[
#    float, int, bool]:

def run(agent_class, agent_config_file_path: Path, carla_config_file_path: Path,
        num_laps: int = 2) -> Tuple[float, int, int]:
    """
    Run the agent along the track and produce a score based on certain metrics
    Args:
        num_laps: int number of laps that the agent should run
        agent_class: the participant's agent
        agent_config_file_path: agent configuration path
        carla_config_file_path: carla configuration path
    Returns:
        float between 0 - 1 representing scores
    """

    agent_config: AgentConfig = AgentConfig.parse_file(agent_config_file_path)
    carla_config = CarlaConfig.parse_file(carla_config_file_path)
    #run_sim
	#"""
    #Pit Stop:
    #    Use different kinds of 'set' functions at PitStop to tune/fix your own car!
    #"""
    pitstop = PitStop(carla_config, agent_config)
	#pitstop: object = PitStop(carla_config, agent_config)
    pitstop.set_carla_version(version = "0.9.12")
    pitstop.set_carla_sync_mode(True)
    pitstop.set_autopilot_mode(True)
    #pitstop.set_car_color(CarlaCarColor(r = 0,g = 0,b = 255,a = 255))
    pitstop.set_num_laps(num=2)
    pitstop.set_output_data_folder_path("./data/output")
    pitstop.set_output_data_file_name(time.strftime("%Y%m%d-%H%M%S-") + "map-waypoints")
    pitstop.set_max_speed(speed = 100)
    pitstop.set_target_speed(speed = 40)
    print(agent_config.target_speed, " target speed")
    #print(agent_config. , " target speed")
    #print(pitstop)
    pitstop.set_steering_boundary(boundary = (-1.0, 1.0))
    pitstop.set_throttle_boundary(boundary = (0, 1))

    pitstop.set_waypoints_look_ahead_values(values={
                                                    "40": 10,
                                                    "60": 20,
                                                    "80": 30,
                                                    "100": 50,
                                                    "110": 55})
    pid_value = {
                    "longitudinal_controller": {
                        "40": {
                            "Kp": 0.8,
                            "Kd": 0.2,
                            "Ki": 0
                        },
                        "60": {
                            "Kp": 0.7,
                            "Kd": 0.2,
                            "Ki": 0
                        },
                        "80": {
                            "Kp": 0.5,
                            "Kd": 0.15,
                            "Ki": 0.05
                        },
                        "100": {
                            "Kp": 0.5,
                            "Kd": 0.1,
                            "Ki": 0
                        },
						"120": {
                			"Kp": 0.2,
                			"Kd": 0.1,
                			"Ki": 0.1
                    	}
					},	
                    "latitudinal_controller": {
			
                        "60": {
                            "Kp": 0.8,
                            "Kd": 0.1,
                            "Ki": 0.2
                        },
                        "80": {
                            "Kp": 0.6,
                            "Kd": 0.2,
                            "Ki": 0.1
                        },
                        "100": {
                            "Kp": 0.5,
                            "Kd": 0.2,
                            "Ki": 0.1
                        },
                        "120": {
                            "Kp": 0.4,
                            "Kd": 0.2,
                            "Ki": 0.2
                        }
                    }
    }
    pitstop.set_pid_values(pid_value)

    """Passing configurations to Carla and Agent"""
    #carla_runner = CarlaRunner(carla_settings=carla_config, # ROAR Academy: fine
    #                           agent_settings=agent_config, # ROAR Academy: fine
    #                           npc_agent_class=PurePursuitAgent) 
    # hard code agent config such that it reflect competition requirements
    agent_config.num_laps = num_laps
    carla_runner = CarlaRunner(carla_settings=carla_config,
                               agent_settings=agent_config,
                               npc_agent_class=PurePursuitAgent,
                               competition_mode=True,
							   start_bbox=np.array([-815, 20, -760, -770, 120, -600]),
                               lap_count=num_laps)
    try:
        my_vehicle = carla_runner.set_carla_world()
        agent = agent_class(vehicle=my_vehicle, agent_settings=agent_config)
        carla_runner.start_game_loop(agent=agent, use_manual_control=False)
        return compute_score(carla_runner)
    except Exception as e:
        print(f"something bad happened during initialization: {e}")
        carla_runner.on_finish()
        logging.error(f"{e}. Might be a good idea to restart Server")
        return 0, 0, False


def suppress_warnings():
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s '
                               '- %(message)s',
                        level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.simplefilter("ignore")
    np.set_printoptions(suppress=True)


def main():
    suppress_warnings()
    agent_class = PIDAgent
    num_trials = 5
    total_score = 0
    num_laps = 2
    table = PrettyTable()
    table.field_names = ["time_elapsed (sec)", "num_collisions", "laps completed"]
    for i in range(num_trials):
        scores = run(agent_class=agent_class,
                     agent_config_file_path=Path("./ROAR/configurations/carla/carla_agent_configuration.json"),
					 #agent_config_file_path=Path("./ROAR/configurations/carla/agent_configuration.json"),
                     carla_config_file_path=Path("./ROAR/configurations/configuration.json"),
					 #agent_config_file_path=Path("./ROAR_Sim/configurations/agent_configuration.json"),
                     #carla_config_file_path=Path("./ROAR_Sim/configurations/configuration.json"),
                     num_laps=num_laps)
        table.add_row(scores)
    print(table)


if __name__ == "__main__":
    main()

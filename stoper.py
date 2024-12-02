import time
from typing import List

class stoper:
    def __init__(self):
        self.laps: List[tuple[float, str]] = []

    def start(self):
        self.laps = []
        self.laps.append((time.time(), 'start'))
    
    def lap(self, name = ''):
        self.laps.append((time.time(), name))

    def get_data(self) -> List[tuple[float, str]]:
        return self.laps
    
    def get_data_str(self) -> str:
        data_str = ''
        prev_time = None
        for lap in self.laps:
            current_time = lap[0]  # Get time directly from tuple
            name = lap[1]          # Get name directly from tuple
            
            if prev_time is None:
                time_diff = 0
            else:
                time_diff = current_time - prev_time
                
            data_str += f"Time: {current_time:.3f}, Diff: {time_diff:.3f}, Name: {name}\n"
            prev_time = current_time
        data_str += f"Total: {(self.laps[len(self.laps)-1][0] - self.laps[0][0]):.3f}"
        return data_str
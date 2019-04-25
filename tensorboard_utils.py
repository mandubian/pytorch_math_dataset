from tensorboardX import SummaryWriter
from pathlib import Path
import datetime
            
from tensorboard.backend.event_processing import event_accumulator

   
def tensorboard_event_accumulator(
    file,
    loaded_scalars=0, # load all scalars by default
    loaded_images=4, # load 4 images by default
    loaded_compressed_histograms=500, # load one histogram by default
    loaded_histograms=1, # load one histogram by default
    loaded_audio=4, # loads 4 audio by default
):
    ea = event_accumulator.EventAccumulator(
        file,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: loaded_compressed_histograms,
            event_accumulator.IMAGES: loaded_images,
            event_accumulator.AUDIO: loaded_audio,
            event_accumulator.SCALARS: loaded_scalars,
            event_accumulator.HISTOGRAMS: loaded_histograms,
        }
    )
    ea.Reload()
    return ea


class Tensorboard:
    def __init__(
        self,
        experiment_id,
        output_dir="./runs",
        unique_name=None,
    ):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        if unique_name is None:
            unique_name = datetime.datetime.now().isoformat(timespec="seconds")
        self.path = self.output_dir / f"{experiment_id}_{unique_name}"
        print(f"Writing TensorBoard events locally to {self.path}")
        self.writers = {}

    def _get_writer(self, group: str=""):
        if group not in self.writers:
            print(
                f"Adding group {group} to writers ({self.writers.keys()})"
            )
            self.writers[group] = SummaryWriter(f"{str(self.path)}_{group}")
        return self.writers[group]
    
    def add_scalars(self, metrics: dict, global_step: int, group=None, sub_group=""):
        for key, val in metrics.items():
            cur_name = "/".join([sub_group, key])
            self._get_writer(group).add_scalar(cur_name, val, global_step)

import logging

from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from codecarbon.emissions_tracker import BaseEmissionsTracker

from mlcost import utils
from mlcost import learn


def start_benchmark(online=False, save_to_file=False, name="") -> BaseEmissionsTracker:
    """Start emission and time tracking."""
    if online:
        tracker = EmissionsTracker(project_name=name, save_to_file=save_to_file,
                                   tracking_mode="process", log_level="error")
    else:
        tracker = OfflineEmissionsTracker(country_iso_code="ESP" , project_name=name, 
                                          save_to_file=save_to_file, tracking_mode="process",
                                          log_level="error")

    tracker.start()
    return tracker


def stop_benchmark(em_tracker: BaseEmissionsTracker):
    """Stop emission and time tracking."""
    em_tracker.stop()
    return em_tracker.final_emissions_data
    

def main(dataset, labels, test, separator, codecarbon_file, cross_validate, online, log, no_header, openml):
    utils.print_computer_info()

    trainer = learn.Trainer(dataset, test, labels, cross_validate, separator, no_header, openml)
    trainer.clean_data(log_output=True)
    
    try:
        for name, model in learn.MODEL_TYPES.items():
            filename = "_".join([codecarbon_file, name]) if codecarbon_file else ""
            em_tracker = start_benchmark(online, bool(codecarbon_file), filename)
            if (cross_validate <= 1):
                predictions = trainer.train(model)
                emission = stop_benchmark(em_tracker)
                score = trainer.score(predictions)
            else:
                score = trainer.cross_score(model)
                emission = stop_benchmark(em_tracker)

            utils.print_output(name, score, emission.duration, emission.emissions, emission.energy_consumed)

            if log:
                utils.log_to_file(trainer.name, score, emission, name)
    except KeyboardInterrupt:
        pass

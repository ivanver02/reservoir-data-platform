# Importing modules
from backend.data.transform.water import etl_pipeline_water
from backend.data.transform.reservoir import etl_pipeline_reservoir
from backend.data.transform.detailed_reservoir import etl_pipeline_detailed_reservoir
from backend.data.transform.merges  import etl_pipeline_reservoirs_merged


def run_all_ETL():
    """
    Main function to run the ETL pipeline.
    """
    etl_pipeline_water()
    etl_pipeline_reservoir()
    etl_pipeline_detailed_reservoir()
    etl_pipeline_reservoirs_merged()

if __name__ == "__main__":
    run_all_ETL()
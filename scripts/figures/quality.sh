python -m vis.quality \
       --name quality_test \
       --start_path log/NeurIPS3/macaw_vel_start.3/tb/events.out.tfevents.1590910906.iris3.stanford.edu.61694.0 \
       --middle_path log/NeurIPS3/macaw_vel_middle.3/tb/events.out.tfevents.1590910906.iris3.stanford.edu.61695.0 \
       --end_path log/NeurIPS3/macaw_vel_end.3/tb/events.out.tfevents.1590910906.iris3.stanford.edu.61696.0 \
       --maml_start_path log/NeurIPS3/mamlawr_vel_start/tb/events.out.tfevents.1590781779.iris1.stanford.edu.30291.0 \
       --maml_middle_path log/NeurIPS3/mamlawr_vel_middle/tb/events.out.tfevents.1590781779.iris1.stanford.edu.30292.0 \
       --maml_end_path log/NeurIPS3/mamlawr_vel_end/tb/events.out.tfevents.1590781779.iris1.stanford.edu.30293.0 \
       --terminate 200000

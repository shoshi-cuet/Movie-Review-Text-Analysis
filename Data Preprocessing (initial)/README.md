1. To combined and to convert into csv format all json files from "Original Data" folder run `python convert-and-combined.py` . You will get the combined_data.csv in "Combined Data" folder. With that combined data we will work further all our sub section task.

2. run this command for copy combined data into ubuntu `scp -i .ssh/DIS_key combined_data.csv ubuntu@152.94.170.149:/home/ubuntu/data` .For putting this combined data into hadoop run `put_into_hadoop.sh`

3. For our testing and to run locally we will also create a test_combined_data.csv in "Test Data" folder from a single json using step 1 (edit input output folder location in the .py file) and by step 2 copy it into hadoop file system.

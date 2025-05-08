import os
import io
import boto3
import logging
from botocore.exceptions import ClientError
import xarray as xr                 
import matplotlib.pyplot as plt
import numpy as np


from credentials_buckets import S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL

def read_file(s3, file_name, bucket):
    """Upload a file to an S3 bucket
    :param s3: Initialized S3 client object
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :return: object if file was uploaded, else False
    """
    try:
        #with open(file_name, "rb") as f:
        obj = s3.get_object(Bucket=bucket, Key=file_name)
        #print(obj)
        myObject = obj['Body'].read()
    except ClientError as e:
        logging.error(e)
        return None
    return myObject


# Initialize the S3 client
s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY
)

# List the objects in our bucket
response = s3.list_objects(Bucket=S3_BUCKET_NAME)
for item in response['Contents']:
    print(item['Key'])

#Directory with the data to uplad
years = [2013]

month_start = 4 #April
month_end = 9 #Septeber
months = range(month_start,month_end+1)

days = range(1,32)

hour_start = '00' #UTC (included)
hour_end = '24' #UTC (not included,
hours = range(int(hour_start),int(hour_end))

path_dir = f"output/data/timeseries_crops"
basename = "MSG_timeseries"

# Define your range limits
#TODO fix this in case more than one IR variable is included
value_min = 200.0  # Example minimum value
value_max = 300.0   # Example maximum value
 
x_pixel = 100 
y_pixel = 100 

cloud_prm = 'IR_108' # IR_108, WV_062, cma

n_crops = 4

#output_path =  f'/work/dcorradi/crops/{cloud_prm_str}_{years_str}_{x_pixel}x{y_pixel}_{domain_name}_{cropping_strategy}/'
outpath = f'/data1/crops/random_walk_frames/{cloud_prm}/train_256'
os.makedirs(outpath, exist_ok=True)

for year in years:
    for month in months:
        month = f"{month:02d}"
        for day in days:
            day = f"{day:02d}"
            for hour in hours:
                hour = f"{hour:02d}"
                # loop over 15 minutes files
                for minutes in [0, 15, 30, 45]:
                    minutes = f"{minutes:02d}"
                    #loop over crop numbers
                    for crop_number in range(n_crops):
                        file = f"{path_dir}/{year:04d}/{month}/{day}/{basename}_{year:04d}-{month}-{day}_{hour}{minutes}_crop{crop_number}.nc"
                        print(file)
                        
                        #try to read the object from the bucket and if not exists, skip it
                        try:
                            #Read file from the bucket
                            my_obj = read_file(s3, file, S3_BUCKET_NAME)
                            
                            #print(f"File {file} found in bucket {S3_BUCKET_NAME}.")
                            #exit()
                        except ClientError as e:
                            logging.error(e)
                            print(f"File {file} not found in bucket {S3_BUCKET_NAME}.")
                            continue
                        
                        if my_obj is not None:
                            print(f"File {file} found in bucket {S3_BUCKET_NAME}.")
                            
                            ds = xr.open_dataset(io.BytesIO(my_obj))
                            print(ds)
                            
                            #select only variable of interest
                            ds = ds[cloud_prm] 

                            print(ds)

                            # Loop over time steps
                            for i, t in enumerate(ds.time.values):
                                frame = ds.sel(time=t).values  # 2D array (lat, lon)
                                
                                # Plot without axis mainting aspect rati
                                fig, ax = plt.subplots(figsize=(x_pixel, y_pixel), dpi=1)
                                frame = np.flipud(frame)
                                ax.imshow(frame, cmap='gray', vmin=value_min, vmax=value_max)
                                ax.axis('off')  # Remove axis ticks and labels
                                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                                # Save to PNG
                                timestamp = str(np.datetime64(t)).replace(':', '').replace('-', '')
                                dic_name = file.split('/')[-1].split('.')[0]
                                os.makedirs(f'{outpath}/{dic_name}', exist_ok=True)

                                filename = f'{outpath}/{dic_name}/{i}.png'
                                plt.savefig(filename)
                                plt.close()

                                print( f' Saved {outpath}/{filename}/{cloud_prm}_{timestamp}.png')

                                        


                            
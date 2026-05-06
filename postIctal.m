clc;
clear;

aa = dir('*.edf');
filename = aa.name;

[hdr, xx] = edfread(filename);
[data, header, cfg] = lab_read_edf(filename);
clear data;

fs_original = round(header.hdr.numbersperrecord(1) / header.hdr.duration);

[p, f, ~] = fileparts(filename);
patient_name = f;

ECG_raw = xx(14,:);

%% ---------------- FILTERING ----------------

f0 = 50; Q = 30;
[b, a] = iirnotch(f0/(fs_original/2), f0/(fs_original*Q));
ECG_raw = filtfilt(b, a, ECG_raw);

[b, a] = butter(4, 0.5/(fs_original/2), 'high');
ECG_raw = filtfilt(b, a, ECG_raw);

[b, a] = butter(4, 20/(fs_original/2), 'low');
ECG_raw = filtfilt(b, a, ECG_raw);

%% ---------------- RESAMPLE ----------------

fs = 250;
ECG_final = resample(ECG_raw, fs, fs_original);

%% ---------------- BASELINE ----------------

ECG_final = ECG_final - mean(ECG_final(1:5000));

%% ---------------- ONSET TIME ----------------

timing_table = readtable('Timings.xlsx', 'Sheet', 'Sheet1');
onset_cell = timing_table{1,2};

if iscell(onset_cell)
    onset_str = onset_cell{1};
else
    onset_str = onset_cell;
end

onset_str = strrep(char(onset_str), '"','');

tOnset = sscanf(onset_str, '%d:%d:%d');
onset_in_seconds = tOnset(1)*3600 + tOnset(2)*60 + tOnset(3);

tStart = sscanf(strrep(hdr.starttime,'.',':'), '%d:%d:%d');
start_in_seconds = tStart(1)*3600 + tStart(2)*60 + tStart(3);

if onset_in_seconds < start_in_seconds
    actual_onset = (24*3600 - start_in_seconds) + onset_in_seconds;
else
    actual_onset = onset_in_seconds - start_in_seconds;
end

%% ---------------- POSTICTAL WINDOW ----------------
% 5 min after onset → take enough buffer

post_start = round((actual_onset + 5*60) * fs);
post_end   = round((actual_onset + 10*60) * fs);

max_index = length(ECG_final);

if post_start >= max_index
    error("Postictal start exceeds signal length.");
end

if post_end > max_index
    post_end = max_index;
end

ECG_post = ECG_final(post_start:post_end);

%% ---------------- FIXED 18 × 10s CHUNKS ----------------

fs_chunk = 10 * fs;   % 10 seconds
num_chunks = 18;
required_samples = num_chunks * fs_chunk;

if length(ECG_post) < required_samples
    error("Not enough postictal data for 18 chunks of 10 seconds.");
end

ECG_post = ECG_post(1:required_samples);

matrix_postictal = reshape(ECG_post, fs_chunk, num_chunks)';

% Label = 2
matrix_postictal = [matrix_postictal, 2 * ones(num_chunks,1)];

writematrix(matrix_postictal, 'ECG_chunks_postictal.xlsx');

%% ---------------- HRV FEATURES ----------------

feature_names = {'BPMglobal','BPMlocal','SDNN','rmsRR','NN50',...
'pNN50','IBImean','IBIrange_1','IBIrange_2',...
'SD1','SD2','lfPower','hfPower'};

num_features = numel(feature_names);
hrv_matrix_postictal = zeros(num_chunks, num_features);

for i = 1:num_chunks
    
    sig = matrix_postictal(i,1:end-1);
    
    d = ECG(sig, fs, 100);
    d.init;

    hrv_matrix_postictal(i,:) = [d.BPMglobal, d.BPMlocal, d.SDNN, d.rmsRR, ...
        d.NN50, d.pNN50, d.IBImean, d.IBIrange(1), d.IBIrange(2), ...
        d.SD1, d.SD2, d.lfPower, d.hfPower];
end

% Add label
hrv_matrix_postictal = [hrv_matrix_postictal, 2*ones(num_chunks,1)];

feature_names_with_label = [feature_names, {'Label'}];

xlswrite('HRV_features_postictal.xlsx', feature_names_with_label, 'A1');
xlswrite('HRV_features_postictal.xlsx', hrv_matrix_postictal, 'A2');

%% ---------------- SAVE ----------------

save('postictal_data.mat', 'matrix_postictal', 'hrv_matrix_postictal');

disp('Postictal processing (18 chunks of 10 seconds) completed successfully.');
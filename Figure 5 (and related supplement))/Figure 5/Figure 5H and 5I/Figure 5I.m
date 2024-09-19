Directory = '/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/Data/231002_PNOC_Beh_Sig_updated_excludeforDMS';
% you need to change the directory to your own 

all_evt =dir(fullfile(Directory,'*/*.evtsav'));
all_greenL = dir(fullfile(Directory,'*/*GreenL.mat'));
all_greenR = dir(fullfile(Directory,'*/*Greenr.mat'));
all_RedL = dir(fullfile(Directory,'*/*RedL.mat'));
all_RedR = dir(fullfile(Directory,'*/*RedR.mat'));

green_target = "Sd1_excluded";
for k =1:length(all_evt)
    folderName = all_evt(k).folder;
    if k ==1
        lfp_read2('delshortframes', 'preset',folderName,{all_evt(k).name,all_greenL(k).name, all_greenR(k).name, all_RedL(k).name, all_RedR(k).name});
    else 
        lfp_add('preset',folderName,{all_evt(k).name,all_greenL(k).name, all_greenR(k).name, all_RedL(k).name, all_RedR(k).name}, 'CSC (Matlab - *.MAT)', true, 'filecheck', @last4match);
    end
end
[values_pnoc_contra_DMS, plotdata_pnoc_contra_DMS] = lfp_makepasteup({TrialStart InitOff TurnEnd ROn ROff}, @lfpdispwrapper, [], 1, [], 'avg', 'err2', 'pasteup_opts', {'data', 'autoselect', 'intervals', [0.6,1.5,0.6,0.5]});
[values_Da_in_poc_contra_DMS, plotdata_Da_in_poc_contra_DMS] = lfp_makepasteup({TrialStart InitOff TurnEnd ROn ROff}, @lfpdispwrapper, [], 2, [], 'avg', 'err2', 'pasteup_opts', {'data', 'autoselect', 'intervals', [0.6,1.5,0.6,0.5]});
get_peak_crosscorr_neuron_dopamine(values_pnoc_contra_DMS, values_Da_in_poc_contra_DMS, "Sd1-Da-contralateral-DMS" );

green_target = "Sd1_excluded";
for k =1:length(all_evt)
    folderName = all_evt(k).folder;
    if k ==1
        lfp_read2('delshortframes', 'preset',folderName,{all_evt(k).name,all_greenL(k).name, all_greenR(k).name, all_RedL(k).name, all_RedR(k).name});
    else 
        lfp_add('preset',folderName,{all_evt(k).name,all_greenL(k).name, all_greenR(k).name, all_RedL(k).name, all_RedR(k).name}, 'CSC (Matlab - *.MAT)', true, 'filecheck', @last4match);
    end
end

[values_pnoc_ipsi_DMS, plotdata_pnoc_ipsi_DMS] = lfp_makepasteup({TrialStart InitOff TurnEnd LOn LOff}, @lfpdispwrapper, [], 1, [], 'avg', 'err2', 'pasteup_opts', {'data', 'autoselect', 'intervals', [0.6,1.5,0.6,0.5]});
[values_Da_in_poc_ipsi_DMS, plotdata_Da_in_poc_ipsi_DMS] = lfp_makepasteup({TrialStart InitOff TurnEnd LOn LOff}, @lfpdispwrapper, [], 2, [], 'avg', 'err2', 'pasteup_opts', {'data', 'autoselect', 'intervals', [0.6,1.5,0.6,0.5]});
get_peak_crosscorr_neuron_dopamine(values_pnoc_ipsi_DMS, values_Da_in_poc_ipsi_DMS, "Sd1-Da-ipsilateral-DMS" );

Directory = '/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/Data/231002_PNOC_Beh_Sig_updated_excludeforDLS';
% you need to change the directory to your own 

all_evt =dir(fullfile(Directory,'*/*.evtsav'));
all_greenL = dir(fullfile(Directory,'*/*GreenL.mat'));
all_greenR = dir(fullfile(Directory,'*/*Greenr.mat'));
all_RedL = dir(fullfile(Directory,'*/*RedL.mat'));
all_RedR = dir(fullfile(Directory,'*/*RedR.mat'));

green_target = "Sd1_excluded";
for k =1:length(all_evt)
    folderName = all_evt(k).folder;
    if k ==1
        lfp_read2('delshortframes', 'preset',folderName,{all_evt(k).name,all_greenL(k).name, all_greenR(k).name, all_RedL(k).name, all_RedR(k).name});
    else 
        lfp_add('preset',folderName,{all_evt(k).name,all_greenL(k).name, all_greenR(k).name, all_RedL(k).name, all_RedR(k).name}, 'CSC (Matlab - *.MAT)', true, 'filecheck', @last4match);
    end
end

[values_pnoc_ipsi_DLS, plotdata_pnoc_ipsi_DLS] = lfp_makepasteup({TrialStart InitOff TurnEnd ROn ROff}, @lfpdispwrapper, [], 3, [], 'avg', 'err2', 'pasteup_opts', {'data', 'autoselect', 'intervals', [0.6,1.5,0.6,0.5]});
[values_Da_in_poc_ipsi_DLS, plotdata_DA_in_poc_ipsi_DLS] = lfp_makepasteup({TrialStart InitOff TurnEnd ROn ROff}, @lfpdispwrapper, [], 4, [], 'avg', 'err2', 'pasteup_opts', {'data', 'autoselect', 'intervals', [0.6,1.5,0.6,0.5]});
get_peak_crosscorr_neuron_dopamine(values_pnoc_ipsi_DLS, values_Da_in_poc_ipsi_DLS, "Sd1-Da-ipsilateral-DLS" );

green_target = "Sd1_excluded";
for k =1:length(all_evt)
    folderName = all_evt(k).folder;
    if k ==1
        lfp_read2('delshortframes', 'preset',folderName,{all_evt(k).name,all_greenL(k).name, all_greenR(k).name, all_RedL(k).name, all_RedR(k).name});
    else 
        lfp_add('preset',folderName,{all_evt(k).name,all_greenL(k).name, all_greenR(k).name, all_RedL(k).name, all_RedR(k).name}, 'CSC (Matlab - *.MAT)', true, 'filecheck', @last4match);
    end
end

[values_pnoc_contra_DLS, plotdata_pnoc_contra_DLS] = lfp_makepasteup({TrialStart InitOff TurnEnd LOn LOff}, @lfpdispwrapper, [], 3, [], 'avg', 'err2', 'pasteup_opts', {'data', 'autoselect', 'intervals', [0.6,1.5,0.6,0.5]});
[values_Da_in_poc_contra_DLS, plotdata_DA_in_poc_contra_DLS] = lfp_makepasteup({TrialStart InitOff TurnEnd LOn LOff}, @lfpdispwrapper, [], 4, [], 'avg', 'err2', 'pasteup_opts', {'data', 'autoselect', 'intervals', [0.6,1.5,0.6,0.5]});
get_peak_crosscorr_neuron_dopamine(values_pnoc_contra_DLS, values_Da_in_poc_contra_DLS, "Sd1-Da-contralateral-DLS" );

data_all = {values_pnoc_contra_DMS, plotdata_pnoc_contra_DMS, values_Da_in_poc_contra_DMS, plotdata_Da_in_poc_contra_DMS, values_pnoc_ipsi_DLS, plotdata_pnoc_ipsi_DLS, values_Da_in_poc_ipsi_DLS, plotdata_DA_in_poc_ipsi_DLS, values_pnoc_ipsi_DMS, plotdata_pnoc_ipsi_DMS, values_Da_in_poc_ipsi_DMS, plotdata_Da_in_poc_ipsi_DMS, values_pnoc_contra_DLS, plotdata_pnoc_contra_DLS, values_Da_in_poc_contra_DLS, plotdata_DA_in_poc_contra_DLS};
varNames = {'values_pnoc_contra_DMS', 'plotdata_pnoc_contra_DMS', 'values_Da_in_poc_contra_DMS', 'plotdata_Da_in_poc_contra_DMS', 'values_pnoc_ipsi_DLS', 'plotdata_pnoc_ipsi_DLS', 'values_Da_in_poc_ipsi_DLS', 'plotdata_DA_in_poc_ipsi_DLS', 'values_pnoc_ipsi_DMS', 'plotdata_pnoc_ipsi_DMS', 'values_Da_in_poc_ipsi_DMS', 'plotdata_Da_in_poc_ipsi_DMS', 'values_pnoc_contra_DLS', 'plotdata_pnoc_contra_DLS', 'values_Da_in_poc_contra_DLS', 'plotdata_DA_in_poc_contra_DLS'};

for i = 1:length(varNames)
    varName = varNames{i};  % Get the variable name as a string
    save([varName '.mat'], varName);
end

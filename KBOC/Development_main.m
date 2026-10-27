% % Keystroke Biometric Ongoing Competition (KBOC) is an official competition of 
% % The IEEE Eighth International Conference on Biometrics: Theory, Applications, and Systems (BTAS 2016) 
% % organized by ATVS Biometric Research Group

% % Development code example: this code is an example of the evaluation protocol 
% % used for the competition. In this case, the genuine and impostor samples are known.
% % Samples from 1 to 14 are genuine while 15 to 24 are impostors.
% % This information will be sequestered for the Test set


clear all
db_dir = 'Development_db\'; % the folder Development_db must be in the same folder that this file

%%%
keytimes=[];
keytimes_props=[];

num_users = 10; % Number of users in the development set
num_samples= 24; % Number of samples is 24 per each user
num_training_samples=4; % Number of training samples used for modeling the user's behavior
num_test_samples=20; % Total number of test samples

genuines=[]; %this vector will contain the genuine scores
impostors=[]; %this vector will contain the impsotor scores
for i_user = 1:num_users    
    fprintf(1, 'Calculating scores from User %d...', i_user);
    user_timing=[]; % this variable will contain the training matrix
    for i_sample = 1: num_training_samples        
        file_name = ['D' num2str(i_user,'%.3d') '_' num2str(i_sample,'%.2d') '.txt'];        
        ks = LoadKeySample([db_dir file_name]); % this function is used to extract the information from the database files        
        user_timing(end+1,:)=ks.timing(2:end); % initial time value (time reference always equal to 0) is removed
    end  
       
    %MODELING based on mean and std of the training matrix
    mean_train = mean(user_timing,1);
    sigmas_train = std(user_timing,1);
    threshold_sigmas = .2*mean(sigmas_train);
    sigmas_train(find(sigmas_train<threshold_sigmas))=threshold_sigmas;
    
    %TESTING based on Manhattan distance
    for i_sample = num_training_samples+1:num_test_samples+num_training_samples        
        file_name = ['D' num2str(i_user,'%.3d') '_' num2str(i_sample,'%.2d') '.txt'];        
        ks = LoadKeySample([db_dir file_name]); 
        user_test = ks.timing(2:end); % Samples used for Testing, initial 0 (time reference) is removed
        score = mean((1./sigmas_train).*abs(user_test - mean_train)); %Manhattan distance
        if i_sample<15 %In the development set the genuine and impostor samples are known. This information will remain sequestered for the Test set.
            genuines(end+1)=-score;
        else
            impostors(end+1)=-score;
        end
    end
    fprintf(1, 'OK\n');
end

fprintf(1, '\nTotal number of scores in the experiment: %d\n', length(genuines)+length(impostors));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Performance Evaluation based on Global EER %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
far=[];
frr=[];
limit1=min([genuines impostors]);
limit2=max([genuines impostors]);
step=(limit2-limit1)/10000;
for threshold=limit1-step:step:limit2+step    
    far(end+1)=length(find(impostors>=threshold))/length(impostors);
    frr(end+1)=length(find(genuines<threshold))/length(genuines);    
end
plot([limit1-step:step:limit2+step],far,[limit1-step:step:limit2+step],frr);
diff=abs(far-frr);
[i,j]=min(diff);
title(['EER: ',num2str((far(j)+frr(j))/2)])
ylabel('Error Rate')
xlabel('Classification Threshold')
grid
drawnow

#######script for creating metrics#######
setwd("~/Fall 2024/BigDataBowl/report_scripts/data")
library(tidyverse)
library(mvtnorm)


###read in data
extratest <- read.delim("extratest.txt", header=FALSE)
colnames(extratest)<- c("gameId","playId","Sequence","Object","nflId","X","Y","unique_play")
raw_data_dict <- read.csv("raw_data_dict.csv")
mean_cov_abs <- read.csv("mean_cov_abs.csv", header=T)
plays <- read.csv("plays.csv")
tracking_week_9 <- read.csv("tracking_week_9.csv")
players <- read.csv("players.csv")


#######structure#######
step_match<-as.data.frame(cbind(unique(extratest$unique_play),1:length(unique(extratest$unique_play))))
colnames(step_match)<- c("unique_play","Step")
extratest<- extratest %>%
  left_join(step_match, by = "unique_play")
extratest$Step<- as.numeric(extratest$Step)
playids<- extratest %>%
  distinct(gameId,playId,Step, unique_play)
raw_data_dict_w_playid<- raw_data_dict %>%
  left_join(playids, by = "Step")

######line up play and sequence
extratest$Object<- extratest$Object-1 #so that object is in 0-22
####checking how many sequences are within each play -> 36, 10(2 sec) pre,1 for snap, 25(5 sec) post
extratest %>%
  group_by(unique_play)%>%
  mutate(ran = range(Sequence)[2]-range(Sequence)[1])%>%
  ungroup()%>%
  distinct(ran) 
####transform sequences to be between 0-35
extratest<- extratest %>%
  rename(oldseq = Sequence)%>%
  group_by(unique_play)%>%
  mutate(Sequence = oldseq - min(oldseq))%>%
  ungroup()
######remove presnap sequences and transform sequences to be in 0-25
extratest_post<- extratest %>%
  filter(Sequence>9) %>%
  mutate(Sequence = Sequence-10)



#########################TEAM METRIC#############################

######bring in time to throw, passResult and target location(adjusted for play direction) and subset plays
####subset plays to just be the ones in question (week 9)
plays_test<- plays %>%
  mutate(unique_play = paste0(gameId,"_",playId))%>%
  filter(unique_play %in% extratest$unique_play)
nrow(plays_test) ###545 in week 9

####obtain play direction for each play
playdir<- tracking_week_9 %>%
  distinct(gameId,playId, playDirection) %>%
  mutate(unique_play = paste0(gameId,"_",playId))%>%
  filter(unique_play %in% extratest$unique_play)%>%
  dplyr::select(unique_play,playDirection)
nrow(playdir)  ###545 matches above

#####add in play direction to plays data & adjust targetX if play direction is going left
plays_test<- plays_test %>%
  left_join(playdir, by = c("unique_play"))%>%
  dplyr::select(unique_play,timeToThrow,passResult,targetX,targetY,playDirection) %>%
  mutate(targetX = ifelse(playDirection == "left",120 - targetX, targetX))

####subset to just 3 seconds or less for timeToThrow
quickthrow<- plays_test %>%
  filter(timeToThrow<=3)
sum(is.na(quickthrow$targetX))           #3 nas    


######remove throw aways
quickthrow<- quickthrow %>%
  filter(is.na(targetX)==FALSE) 



######create metric for each play

####join plays data to raw data from model
raw_quick_throw<- raw_data_dict_w_playid %>%
  left_join(quickthrow, by = c("unique_play")) %>%
  filter(is.na(timeToThrow)==FALSE)

###metric creation
raw_quick_throw_metric<- raw_quick_throw %>%
  mutate(timeToThrow = round(timeToThrow*5)) %>% #transform timeToThrow to be in frames (.2 seconds per frame)
  filter(Type != "obs")%>% #remove presnap data
  mutate(type2 = ifelse(Type == "trgt",Type,"pred"))%>% 
  filter(type2 == "pred", 
         Sequence == timeToThrow, #keep only Sequences where the sequence was equal to timeToThrow
         Object<=10) %>% #filter out offense and target rows
  mutate(dist2target = sqrt((targetX-X)^2+(targetY-Y)^2))%>% #calculate distance from each sample to the target of throw
  mutate(lessthan2 = dist2target<=2)%>% #logical for if distance is less than 2 yards
  group_by(Step, Sequence)%>% 
  mutate(proplessthan2 = mean(lessthan2))%>% #calculate proportion for each play/sequence
  ungroup()


###extract play passresult timetothrow and propless
metric_table<- raw_quick_throw_metric %>%
  distinct(unique_play, passResult,proplessthan2, timeToThrow)



#obtain offense and defense teams for each play in question
plays_offdef<- plays %>%
  mutate(unique_play = paste0(gameId,"_",playId))%>%
  filter(unique_play %in% metric_table$unique_play) %>%
  dplyr::select(unique_play,possessionTeam,defensiveTeam)
#join in with the metrics
metric_table<- metric_table %>%
  left_join(plays_offdef, by = "unique_play")

#create rates for offensive and defensive teams
metric_table <- metric_table %>%
  group_by(possessionTeam) %>%
  mutate(off_rate = mean(proplessthan2)) %>%
  ungroup()%>%
  group_by(defensiveTeam) %>%
  mutate(def_rate = mean(proplessthan2))%>%
  ungroup()


#####ranking (focus on defensive)
def_ranks<- metric_table %>%
  distinct(defensiveTeam, def_rate)

###################PLAYER METRIC#####################
######add player name and position
players<- players %>%
  dplyr::select(nflId, position,displayName)

extratest<- extratest %>%
  left_join(players, by = "nflId")


#######subset to just safeties and post snap plays and we only need up to the 5th sequence (1sec)
safeties<- extratest %>%
  filter(position == "SS") %>%
  dplyr::select(unique_play,Sequence,Step,Object,nflId,position,displayName)
raw_safeties<- raw_data_dict %>%
  left_join(safeties, by = c("Step","Sequence","Object"))%>%
  filter(is.na(position) == FALSE)%>%
  filter(Type != "obs", Sequence <= 5)
####merge in mean and cov matrix
raw_safeties<- raw_safeties %>%
  left_join(mean_cov_abs, by = c("Step","Sequence","Object"))
####target likelihood, prediction likelihoods in same column, count if it's greater or less, na for trgt

###create pdf values for each sample and target (real value)
raw_safeties_metric<- raw_safeties %>%
  rowwise() %>%  # Ensure operations are performed row-wise
  mutate(
    pdf = dmvnorm(
      c(X, Y),  # Point (X, Y)
      mean = c(mean_abs_X, mean_abs_Y),  # Mean vector
      sigma = matrix(c(C00, C01, C10, C11), ncol = 2)  # Covariance matrix
    )
  ) %>%
  ungroup()

###create column for target value's pdf at each sequence
raw_safeties_metric<- raw_safeties_metric %>%
  mutate(tgt_pdf = ifelse(Type == "trgt", pdf, -100))%>%
  group_by(Step, Sequence, Object) %>%
  mutate(tgt_pdf = max(tgt_pdf))%>%
  ungroup()

###compare each sample's pdf with the target value's pdf
raw_safeties_metric<- raw_safeties_metric %>%
  mutate(eval = ifelse(Type == "trgt", NA, pdf<=tgt_pdf))%>%
  group_by(Step, Sequence, Object) %>%
  mutate(proplower = mean(eval, na.rm = T))%>% #calculate proportion of samples where the pdf was lower
  ungroup()
####at each sequence,we have a proportion
####look at one sequence (subset down) 2 (.4 sec)
raw_safeties_metric_2<- raw_safeties_metric%>%
  filter(Sequence == 2)
###what does the distribution look like
raw_safeties_metric_2 %>%
  ggplot()+
  geom_histogram(aes(x = proplower))

###create threshold indicating disguise#####

raw_safeties_metric_2<- raw_safeties_metric_2 %>%
  mutate(disguise = proplower<=.1)

###rank players by how often they disguise#####
player_ranks<- raw_safeties_metric_2 %>%
  distinct(unique_play,nflId,displayName,proplower,eval,pdf,disguise) %>%
  group_by(nflId)%>%
  mutate(totaldis = sum(disguise), #total number of disguised plays
         disguise_rate = mean(disguise)) %>% #rate
  distinct(displayName,totaldis,disguise_rate)

View(player_ranks)



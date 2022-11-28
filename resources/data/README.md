# Data

This directory is used to host the data set(s).
Data set(s) are available at the following links:

- [Counsel Chat](https://towardsdatascience.com/counsel-chat-bootstrapping-high-quality-therapy-data-971b419f33da): hosted via [GitHub](https://github.com/nbertagnolli/counsel|-chat)
- [Counseling and Psychotherapy Transcripts: Volume II](https://search.alexanderstreet.com/ctrn/browse/title?showall=true): hosted by [Alexander Street](https://search.alexanderstreet.com) 
- [DailyDialog](https://www.aclweb.org/anthology/I17-1099/): hosted by [ParlAI](https://parl.ai) ([download link](https://parl.ai/downloads/dailydialog/dailydialog.tar.gz))
- [EmpatheticDialogues](https://www.aclweb.org/anthology/P19-1534/): hosted by [ParlAI](https://parl.ai) ([download link](https://parl.ai/downloads/empatheticdialogues/empatheticdialogues.tar.gz))
- [HOPE Dataset](https://dl.acm.org/doi/10.1145/3488560.3498509): hosted by authors with restricted ([request form link](https://docs.google.com/forms/d/e/1FAIpQLSfX_7yzABPtdo5FuhEPw8mosHJmHt|-|-3W6s4nTkL1ot7OCCiA/viewform))
- [IEMOCAP](https://doi.org/10.1007/s10579-008-9076-6): hosted by the [Signal Analysis and Interpretation Laboratory](https://sail.usc.edu) (SAIL) of the [University of Southern California](https://www.usc.edu) (USC) ([request website link](https://sail.usc.edu/iemocap/iemocap_release.htm))
- [Persona-Chat](https://aclanthology.org/P18-1205/): hosted by [ParlAI](https://parl.ai) ([download link](https://parl.ai/downloads/personachat/personachat.tgz))
- [Wizard of Wikipedia](https://arxiv.org/abs/1811.01241): hosted by [ParlAI](https://parl.ai) ([download link](https://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz))

Directory structure:
```
 |- data/
    |- cache/
      |- ...
    |- raw/
      |- Counsel_Chat/
        |- counselchatdata.json
      |- Counseling_and_Psychotherapy_Transcripts_Volume_II/
        |- 00000.txt
        |- 00001.txt
        |- ...
      |- dailydialog/
        |- test.json
        |- train.json
        |- valid.json
      |- empatheticdialogues/
        |- test.csv
        |- train.csv
        |- valid.csv
      |- HOPE_WSDM_2022/
        |- Test/
          |- Copy of 2.csv
          |- Copy of 4.csv
          |- ...
        |- Train/
          |- Copy of 1.csv
          |- Copy of 3.csv
          |- ...
        |- Validation/
          |- Copy of 16.csv
          |- Copy of 19.csv
          |- ...
      |- IEMOCAP_full_release
        |- Documentation
          |- dialog
            |- avi
              |- ...
            |- EmoEvaluation
              |- ...
            |- lab
              |- ...
            |- MOCAP_hand
              |- ...
            |- MOCAP_head
              |- ...
            |- MOCAP_rotated
              |- ...
            |- transcriptions
              |- Ses01F_impro01.txt
              |- Ses01F_impro02.txt
              |- ...
            |- wav
              |- ...
          |- sentences
            |- ForcedAlignment
              |- ...
            |- MOCAP_hand
              |- ...
            |- MOCAP_head
              |- ...
            |- MOCAP_rotated
              |- ...
            |- wav
              |- Ses01F_impro01
                |- Ses01F_impro01_F000.wav
                |- Ses01F_impro01_F001.wav
                |- ...
              |- Ses01F_impro02
              |- ...
        |- Session1
          |- [same]
        |- Session2
          |- [same]
        |- Session3
          |- [same]
        |- Session4
          |- [same]
        |- Session5
          |- [same]
        |- README.txt
      |- personachat/
        |- test_both_original.txt
        |- test_both_revised.txt
        |- ...
      |- wizard_of_wikipedia/
        |- data.json
        |- test_random_split.json
        |- ...
```
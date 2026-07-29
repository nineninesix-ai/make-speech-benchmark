---
dataset_info:
- config_name: en-US
  features:
  - name: utt
    dtype: string
  - name: lang
    dtype: string
  - name: subset
    dtype: string
  - name: source
    dtype: string
  - name: cv_version
    dtype: string
  - name: prompt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: prompt_audio_orig
    dtype: audio
  - name: prompt_text
    dtype: string
  - name: prompt_dur
    dtype: float64
  - name: prompt_sr_orig
    dtype: int64
  - name: prompt_dur_bin
    dtype: string
  - name: prompt_cv_path
    dtype: string
  - name: prompt_split_origin
    dtype: string
  - name: speaker_id
    dtype: string
  - name: speaker_gender
    dtype: string
  - name: speaker_gender_source
    dtype: string
  - name: speaker_age
    dtype: string
  - name: speaker_accent_label
    dtype: string
  - name: speaker_n_in_subset
    dtype: int64
  - name: qc_vad_speech_ratio
    dtype: float64
  - name: qc_lead_sil
    dtype: float64
  - name: qc_trail_sil
    dtype: float64
  - name: qc_clip_rate
    dtype: float64
  - name: qc_bandwidth_hz
    dtype: float64
  - name: qc_spk_centroid_dist
    dtype: float32
  - name: qc_dnsmos_ovrl
    dtype: float32
  - name: qc_dnsmos_sig
    dtype: float32
  - name: qc_dnsmos_bak
    dtype: float32
  - name: text
    dtype: string
  - name: text_norm
    dtype: string
  - name: n_words
    dtype: int64
  - name: n_chars
    dtype: int64
  - name: len_bin
    dtype: string
  - name: phones
    dtype: string
  - name: n_phones
    dtype: int64
  - name: punct_type
    dtype: string
  - name: text_cv_path
    dtype: string
  - name: text_split_origin
    dtype: string
  - name: gt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_gt
    dtype: bool
  - name: gt_dur
    dtype: float64
  - name: gt_speaker_id
    dtype: string
  - name: gt_same_speaker
    dtype: bool
  - name: gt_gender
    dtype: string
  - name: sim_ref_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_sim_ref
    dtype: bool
  - name: sim_ref_dur
    dtype: float64
  - name: sim_ref_text
    dtype: string
  - name: category
    dtype: string
  - name: subcategory
    dtype: string
  - name: tags
    list: string
  - name: difficulty
    dtype: int64
  - name: has_digit
    dtype: bool
  - name: has_abbrev
    dtype: bool
  - name: has_foreign
    dtype: bool
  - name: notes
    dtype: string
  - name: sim_ref_dur_bin
    dtype: string
  - name: anchor_asr
    dtype: string
  - name: anchor_wer
    dtype: float32
  - name: anchor_cer
    dtype: float32
  - name: anchor_hyp
    dtype: string
  - name: anchor_subs
    dtype: int32
  - name: anchor_dels
    dtype: int32
  - name: anchor_ins
    dtype: int32
  - name: anchor_n_ref_words
    dtype: int32
  - name: anchor_cer_err
    dtype: int32
  - name: anchor_n_ref_chars
    dtype: int32
  - name: anchor_wer_mms
    dtype: float32
  - name: anchor_cer_mms
    dtype: float32
  - name: anchor_subs_mms
    dtype: int32
  - name: anchor_dels_mms
    dtype: int32
  - name: anchor_ins_mms
    dtype: int32
  - name: anchor_cer_err_mms
    dtype: int32
  - name: anchor_wer_scribe
    dtype: float32
  - name: anchor_cer_scribe
    dtype: float32
  - name: anchor_subs_scribe
    dtype: int32
  - name: anchor_dels_scribe
    dtype: int32
  - name: anchor_ins_scribe
    dtype: int32
  - name: anchor_cer_err_scribe
    dtype: int32
  - name: anchor_hyp_scribe
    dtype: string
  - name: anchor_sim_wavlm_sv
    dtype: float32
  - name: anchor_sim_wavlm_ft
    dtype: float32
  - name: anchor_sim_ecapa
    dtype: float32
  - name: gt_qc_dnsmos_ovrl
    dtype: float32
  - name: gt_qc_dnsmos_sig
    dtype: float32
  - name: gt_qc_dnsmos_bak
    dtype: float32
  - name: gt_qc_clip_rate
    dtype: float32
  - name: gt_qc_hf_energy_db
    dtype: float32
  - name: gt_qc_vad_speech_ratio
    dtype: float32
  - name: gt_qc_lead_sil
    dtype: float32
  - name: gt_qc_trail_sil
    dtype: float32
  - name: gt_qc_fails_prompt_qc
    dtype: bool
  - name: simref_qc_dnsmos_ovrl
    dtype: float32
  - name: simref_qc_dnsmos_sig
    dtype: float32
  - name: simref_qc_dnsmos_bak
    dtype: float32
  - name: simref_qc_clip_rate
    dtype: float32
  - name: simref_qc_hf_energy_db
    dtype: float32
  - name: simref_qc_vad_speech_ratio
    dtype: float32
  - name: simref_qc_lead_sil
    dtype: float32
  - name: simref_qc_trail_sil
    dtype: float32
  - name: simref_qc_fails_prompt_qc
    dtype: bool
  - name: qc_hf_energy_db
    dtype: float64
  splits:
  - name: main
    num_bytes: 865319754
    num_examples: 1500
  download_size: 865319754
  dataset_size: 865319754
- config_name: es-ES
  features:
  - name: utt
    dtype: string
  - name: lang
    dtype: string
  - name: subset
    dtype: string
  - name: source
    dtype: string
  - name: cv_version
    dtype: string
  - name: prompt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: prompt_audio_orig
    dtype: audio
  - name: prompt_text
    dtype: string
  - name: prompt_dur
    dtype: float64
  - name: prompt_sr_orig
    dtype: int64
  - name: prompt_dur_bin
    dtype: string
  - name: prompt_cv_path
    dtype: string
  - name: prompt_split_origin
    dtype: string
  - name: speaker_id
    dtype: string
  - name: speaker_gender
    dtype: string
  - name: speaker_gender_source
    dtype: string
  - name: speaker_age
    dtype: string
  - name: speaker_accent_label
    dtype: string
  - name: speaker_n_in_subset
    dtype: int64
  - name: qc_vad_speech_ratio
    dtype: float64
  - name: qc_lead_sil
    dtype: float64
  - name: qc_trail_sil
    dtype: float64
  - name: qc_clip_rate
    dtype: float64
  - name: qc_bandwidth_hz
    dtype: float64
  - name: qc_spk_centroid_dist
    dtype: float32
  - name: qc_dnsmos_ovrl
    dtype: float32
  - name: qc_dnsmos_sig
    dtype: float32
  - name: qc_dnsmos_bak
    dtype: float32
  - name: text
    dtype: string
  - name: text_norm
    dtype: string
  - name: n_words
    dtype: int64
  - name: n_chars
    dtype: int64
  - name: len_bin
    dtype: string
  - name: phones
    dtype: string
  - name: n_phones
    dtype: int64
  - name: punct_type
    dtype: string
  - name: text_cv_path
    dtype: string
  - name: text_split_origin
    dtype: string
  - name: gt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_gt
    dtype: bool
  - name: gt_dur
    dtype: float64
  - name: gt_speaker_id
    dtype: string
  - name: gt_same_speaker
    dtype: bool
  - name: gt_gender
    dtype: string
  - name: sim_ref_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_sim_ref
    dtype: bool
  - name: sim_ref_dur
    dtype: float64
  - name: sim_ref_text
    dtype: string
  - name: category
    dtype: string
  - name: subcategory
    dtype: string
  - name: tags
    list: string
  - name: difficulty
    dtype: int64
  - name: has_digit
    dtype: bool
  - name: has_abbrev
    dtype: bool
  - name: has_foreign
    dtype: bool
  - name: notes
    dtype: string
  - name: sim_ref_dur_bin
    dtype: string
  - name: anchor_asr
    dtype: string
  - name: anchor_wer
    dtype: float32
  - name: anchor_cer
    dtype: float32
  - name: anchor_hyp
    dtype: string
  - name: anchor_subs
    dtype: int32
  - name: anchor_dels
    dtype: int32
  - name: anchor_ins
    dtype: int32
  - name: anchor_n_ref_words
    dtype: int32
  - name: anchor_cer_err
    dtype: int32
  - name: anchor_n_ref_chars
    dtype: int32
  - name: anchor_wer_mms
    dtype: float32
  - name: anchor_cer_mms
    dtype: float32
  - name: anchor_subs_mms
    dtype: int32
  - name: anchor_dels_mms
    dtype: int32
  - name: anchor_ins_mms
    dtype: int32
  - name: anchor_cer_err_mms
    dtype: int32
  - name: anchor_wer_scribe
    dtype: float32
  - name: anchor_cer_scribe
    dtype: float32
  - name: anchor_subs_scribe
    dtype: int32
  - name: anchor_dels_scribe
    dtype: int32
  - name: anchor_ins_scribe
    dtype: int32
  - name: anchor_cer_err_scribe
    dtype: int32
  - name: anchor_hyp_scribe
    dtype: string
  - name: anchor_sim_wavlm_sv
    dtype: float32
  - name: anchor_sim_wavlm_ft
    dtype: float32
  - name: anchor_sim_ecapa
    dtype: float32
  - name: gt_qc_dnsmos_ovrl
    dtype: float32
  - name: gt_qc_dnsmos_sig
    dtype: float32
  - name: gt_qc_dnsmos_bak
    dtype: float32
  - name: gt_qc_clip_rate
    dtype: float32
  - name: gt_qc_hf_energy_db
    dtype: float32
  - name: gt_qc_vad_speech_ratio
    dtype: float32
  - name: gt_qc_lead_sil
    dtype: float32
  - name: gt_qc_trail_sil
    dtype: float32
  - name: gt_qc_fails_prompt_qc
    dtype: bool
  - name: simref_qc_dnsmos_ovrl
    dtype: float32
  - name: simref_qc_dnsmos_sig
    dtype: float32
  - name: simref_qc_dnsmos_bak
    dtype: float32
  - name: simref_qc_clip_rate
    dtype: float32
  - name: simref_qc_hf_energy_db
    dtype: float32
  - name: simref_qc_vad_speech_ratio
    dtype: float32
  - name: simref_qc_lead_sil
    dtype: float32
  - name: simref_qc_trail_sil
    dtype: float32
  - name: simref_qc_fails_prompt_qc
    dtype: bool
  - name: qc_hf_energy_db
    dtype: float64
  splits:
  - name: main
    num_bytes: 888574516
    num_examples: 1500
  download_size: 888574516
  dataset_size: 888574516
- config_name: es-MX
  features:
  - name: utt
    dtype: string
  - name: lang
    dtype: string
  - name: subset
    dtype: string
  - name: source
    dtype: string
  - name: cv_version
    dtype: string
  - name: prompt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: prompt_audio_orig
    dtype: audio
  - name: prompt_text
    dtype: string
  - name: prompt_dur
    dtype: float64
  - name: prompt_sr_orig
    dtype: int64
  - name: prompt_dur_bin
    dtype: string
  - name: prompt_cv_path
    dtype: string
  - name: prompt_split_origin
    dtype: string
  - name: speaker_id
    dtype: string
  - name: speaker_gender
    dtype: string
  - name: speaker_gender_source
    dtype: string
  - name: speaker_age
    dtype: string
  - name: speaker_accent_label
    dtype: string
  - name: speaker_n_in_subset
    dtype: int64
  - name: qc_vad_speech_ratio
    dtype: float64
  - name: qc_lead_sil
    dtype: float64
  - name: qc_trail_sil
    dtype: float64
  - name: qc_clip_rate
    dtype: float64
  - name: qc_bandwidth_hz
    dtype: float64
  - name: qc_spk_centroid_dist
    dtype: float32
  - name: qc_dnsmos_ovrl
    dtype: float32
  - name: qc_dnsmos_sig
    dtype: float32
  - name: qc_dnsmos_bak
    dtype: float32
  - name: text
    dtype: string
  - name: text_norm
    dtype: string
  - name: n_words
    dtype: int64
  - name: n_chars
    dtype: int64
  - name: len_bin
    dtype: string
  - name: phones
    dtype: string
  - name: n_phones
    dtype: int64
  - name: punct_type
    dtype: string
  - name: text_cv_path
    dtype: string
  - name: text_split_origin
    dtype: string
  - name: gt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_gt
    dtype: bool
  - name: gt_dur
    dtype: float64
  - name: gt_speaker_id
    dtype: string
  - name: gt_same_speaker
    dtype: bool
  - name: gt_gender
    dtype: string
  - name: sim_ref_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_sim_ref
    dtype: bool
  - name: sim_ref_dur
    dtype: float64
  - name: sim_ref_text
    dtype: string
  - name: category
    dtype: string
  - name: subcategory
    dtype: string
  - name: tags
    list: string
  - name: difficulty
    dtype: int64
  - name: has_digit
    dtype: bool
  - name: has_abbrev
    dtype: bool
  - name: has_foreign
    dtype: bool
  - name: notes
    dtype: string
  - name: sim_ref_dur_bin
    dtype: string
  - name: anchor_asr
    dtype: string
  - name: anchor_wer
    dtype: float32
  - name: anchor_cer
    dtype: float32
  - name: anchor_hyp
    dtype: string
  - name: anchor_subs
    dtype: int32
  - name: anchor_dels
    dtype: int32
  - name: anchor_ins
    dtype: int32
  - name: anchor_n_ref_words
    dtype: int32
  - name: anchor_cer_err
    dtype: int32
  - name: anchor_n_ref_chars
    dtype: int32
  - name: anchor_wer_mms
    dtype: float32
  - name: anchor_cer_mms
    dtype: float32
  - name: anchor_subs_mms
    dtype: int32
  - name: anchor_dels_mms
    dtype: int32
  - name: anchor_ins_mms
    dtype: int32
  - name: anchor_cer_err_mms
    dtype: int32
  - name: anchor_wer_scribe
    dtype: float32
  - name: anchor_cer_scribe
    dtype: float32
  - name: anchor_subs_scribe
    dtype: int32
  - name: anchor_dels_scribe
    dtype: int32
  - name: anchor_ins_scribe
    dtype: int32
  - name: anchor_cer_err_scribe
    dtype: int32
  - name: anchor_hyp_scribe
    dtype: string
  - name: anchor_sim_wavlm_sv
    dtype: float32
  - name: anchor_sim_wavlm_ft
    dtype: float32
  - name: anchor_sim_ecapa
    dtype: float32
  - name: gt_qc_dnsmos_ovrl
    dtype: float32
  - name: gt_qc_dnsmos_sig
    dtype: float32
  - name: gt_qc_dnsmos_bak
    dtype: float32
  - name: gt_qc_clip_rate
    dtype: float32
  - name: gt_qc_hf_energy_db
    dtype: float32
  - name: gt_qc_vad_speech_ratio
    dtype: float32
  - name: gt_qc_lead_sil
    dtype: float32
  - name: gt_qc_trail_sil
    dtype: float32
  - name: gt_qc_fails_prompt_qc
    dtype: bool
  - name: simref_qc_dnsmos_ovrl
    dtype: float32
  - name: simref_qc_dnsmos_sig
    dtype: float32
  - name: simref_qc_dnsmos_bak
    dtype: float32
  - name: simref_qc_clip_rate
    dtype: float32
  - name: simref_qc_hf_energy_db
    dtype: float32
  - name: simref_qc_vad_speech_ratio
    dtype: float32
  - name: simref_qc_lead_sil
    dtype: float32
  - name: simref_qc_trail_sil
    dtype: float32
  - name: simref_qc_fails_prompt_qc
    dtype: bool
  - name: qc_hf_energy_db
    dtype: float64
  splits:
  - name: main
    num_bytes: 921596856
    num_examples: 1500
  download_size: 921596856
  dataset_size: 921596856
- config_name: nl-NL
  features:
  - name: utt
    dtype: string
  - name: lang
    dtype: string
  - name: subset
    dtype: string
  - name: source
    dtype: string
  - name: cv_version
    dtype: string
  - name: prompt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: prompt_audio_orig
    dtype: audio
  - name: prompt_text
    dtype: string
  - name: prompt_dur
    dtype: float64
  - name: prompt_sr_orig
    dtype: int64
  - name: prompt_dur_bin
    dtype: string
  - name: prompt_cv_path
    dtype: string
  - name: prompt_split_origin
    dtype: string
  - name: speaker_id
    dtype: string
  - name: speaker_gender
    dtype: string
  - name: speaker_gender_source
    dtype: string
  - name: speaker_age
    dtype: string
  - name: speaker_accent_label
    dtype: string
  - name: speaker_n_in_subset
    dtype: int64
  - name: qc_vad_speech_ratio
    dtype: float64
  - name: qc_lead_sil
    dtype: float64
  - name: qc_trail_sil
    dtype: float64
  - name: qc_clip_rate
    dtype: float64
  - name: qc_bandwidth_hz
    dtype: float64
  - name: qc_spk_centroid_dist
    dtype: float32
  - name: qc_dnsmos_ovrl
    dtype: float32
  - name: qc_dnsmos_sig
    dtype: float32
  - name: qc_dnsmos_bak
    dtype: float32
  - name: text
    dtype: string
  - name: text_norm
    dtype: string
  - name: n_words
    dtype: int64
  - name: n_chars
    dtype: int64
  - name: len_bin
    dtype: string
  - name: phones
    dtype: string
  - name: n_phones
    dtype: int64
  - name: punct_type
    dtype: string
  - name: text_cv_path
    dtype: string
  - name: text_split_origin
    dtype: string
  - name: gt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_gt
    dtype: bool
  - name: gt_dur
    dtype: float64
  - name: gt_speaker_id
    dtype: string
  - name: gt_same_speaker
    dtype: bool
  - name: gt_gender
    dtype: string
  - name: sim_ref_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_sim_ref
    dtype: bool
  - name: sim_ref_dur
    dtype: float64
  - name: sim_ref_text
    dtype: string
  - name: category
    dtype: string
  - name: subcategory
    dtype: string
  - name: tags
    list: string
  - name: difficulty
    dtype: int64
  - name: has_digit
    dtype: bool
  - name: has_abbrev
    dtype: bool
  - name: has_foreign
    dtype: bool
  - name: notes
    dtype: string
  - name: sim_ref_dur_bin
    dtype: string
  - name: anchor_asr
    dtype: string
  - name: anchor_wer
    dtype: float32
  - name: anchor_cer
    dtype: float32
  - name: anchor_hyp
    dtype: string
  - name: anchor_subs
    dtype: int32
  - name: anchor_dels
    dtype: int32
  - name: anchor_ins
    dtype: int32
  - name: anchor_n_ref_words
    dtype: int32
  - name: anchor_cer_err
    dtype: int32
  - name: anchor_n_ref_chars
    dtype: int32
  - name: anchor_wer_mms
    dtype: float32
  - name: anchor_cer_mms
    dtype: float32
  - name: anchor_subs_mms
    dtype: int32
  - name: anchor_dels_mms
    dtype: int32
  - name: anchor_ins_mms
    dtype: int32
  - name: anchor_cer_err_mms
    dtype: int32
  - name: anchor_wer_scribe
    dtype: float32
  - name: anchor_cer_scribe
    dtype: float32
  - name: anchor_subs_scribe
    dtype: int32
  - name: anchor_dels_scribe
    dtype: int32
  - name: anchor_ins_scribe
    dtype: int32
  - name: anchor_cer_err_scribe
    dtype: int32
  - name: anchor_hyp_scribe
    dtype: string
  - name: anchor_sim_wavlm_sv
    dtype: float32
  - name: anchor_sim_wavlm_ft
    dtype: float32
  - name: anchor_sim_ecapa
    dtype: float32
  - name: gt_qc_dnsmos_ovrl
    dtype: float32
  - name: gt_qc_dnsmos_sig
    dtype: float32
  - name: gt_qc_dnsmos_bak
    dtype: float32
  - name: gt_qc_clip_rate
    dtype: float32
  - name: gt_qc_hf_energy_db
    dtype: float32
  - name: gt_qc_vad_speech_ratio
    dtype: float32
  - name: gt_qc_lead_sil
    dtype: float32
  - name: gt_qc_trail_sil
    dtype: float32
  - name: gt_qc_fails_prompt_qc
    dtype: bool
  - name: simref_qc_dnsmos_ovrl
    dtype: float32
  - name: simref_qc_dnsmos_sig
    dtype: float32
  - name: simref_qc_dnsmos_bak
    dtype: float32
  - name: simref_qc_clip_rate
    dtype: float32
  - name: simref_qc_hf_energy_db
    dtype: float32
  - name: simref_qc_vad_speech_ratio
    dtype: float32
  - name: simref_qc_lead_sil
    dtype: float32
  - name: simref_qc_trail_sil
    dtype: float32
  - name: simref_qc_fails_prompt_qc
    dtype: bool
  - name: qc_hf_energy_db
    dtype: float64
  splits:
  - name: main
    num_bytes: 724091804
    num_examples: 1500
  download_size: 724091804
  dataset_size: 724091804
- config_name: pt-BR
  features:
  - name: utt
    dtype: string
  - name: lang
    dtype: string
  - name: subset
    dtype: string
  - name: source
    dtype: string
  - name: cv_version
    dtype: string
  - name: prompt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: prompt_audio_orig
    dtype: audio
  - name: prompt_text
    dtype: string
  - name: prompt_dur
    dtype: float64
  - name: prompt_sr_orig
    dtype: int64
  - name: prompt_dur_bin
    dtype: string
  - name: prompt_cv_path
    dtype: string
  - name: prompt_split_origin
    dtype: string
  - name: speaker_id
    dtype: string
  - name: speaker_gender
    dtype: string
  - name: speaker_gender_source
    dtype: string
  - name: speaker_age
    dtype: string
  - name: speaker_accent_label
    dtype: string
  - name: speaker_n_in_subset
    dtype: int64
  - name: qc_vad_speech_ratio
    dtype: float64
  - name: qc_lead_sil
    dtype: float64
  - name: qc_trail_sil
    dtype: float64
  - name: qc_clip_rate
    dtype: float64
  - name: qc_bandwidth_hz
    dtype: float64
  - name: qc_spk_centroid_dist
    dtype: float32
  - name: qc_dnsmos_ovrl
    dtype: float32
  - name: qc_dnsmos_sig
    dtype: float32
  - name: qc_dnsmos_bak
    dtype: float32
  - name: text
    dtype: string
  - name: text_norm
    dtype: string
  - name: n_words
    dtype: int64
  - name: n_chars
    dtype: int64
  - name: len_bin
    dtype: string
  - name: phones
    dtype: string
  - name: n_phones
    dtype: int64
  - name: punct_type
    dtype: string
  - name: text_cv_path
    dtype: string
  - name: text_split_origin
    dtype: string
  - name: gt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_gt
    dtype: bool
  - name: gt_dur
    dtype: float64
  - name: gt_speaker_id
    dtype: string
  - name: gt_same_speaker
    dtype: bool
  - name: gt_gender
    dtype: string
  - name: sim_ref_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_sim_ref
    dtype: bool
  - name: sim_ref_dur
    dtype: float64
  - name: sim_ref_text
    dtype: string
  - name: category
    dtype: string
  - name: subcategory
    dtype: string
  - name: tags
    list: string
  - name: difficulty
    dtype: int64
  - name: has_digit
    dtype: bool
  - name: has_abbrev
    dtype: bool
  - name: has_foreign
    dtype: bool
  - name: notes
    dtype: string
  - name: sim_ref_dur_bin
    dtype: string
  - name: anchor_asr
    dtype: string
  - name: anchor_wer
    dtype: float32
  - name: anchor_cer
    dtype: float32
  - name: anchor_hyp
    dtype: string
  - name: anchor_subs
    dtype: int32
  - name: anchor_dels
    dtype: int32
  - name: anchor_ins
    dtype: int32
  - name: anchor_n_ref_words
    dtype: int32
  - name: anchor_cer_err
    dtype: int32
  - name: anchor_n_ref_chars
    dtype: int32
  - name: anchor_wer_mms
    dtype: float32
  - name: anchor_cer_mms
    dtype: float32
  - name: anchor_subs_mms
    dtype: int32
  - name: anchor_dels_mms
    dtype: int32
  - name: anchor_ins_mms
    dtype: int32
  - name: anchor_cer_err_mms
    dtype: int32
  - name: anchor_wer_scribe
    dtype: float32
  - name: anchor_cer_scribe
    dtype: float32
  - name: anchor_subs_scribe
    dtype: int32
  - name: anchor_dels_scribe
    dtype: int32
  - name: anchor_ins_scribe
    dtype: int32
  - name: anchor_cer_err_scribe
    dtype: int32
  - name: anchor_hyp_scribe
    dtype: string
  - name: anchor_sim_wavlm_sv
    dtype: float32
  - name: anchor_sim_wavlm_ft
    dtype: float32
  - name: anchor_sim_ecapa
    dtype: float32
  - name: gt_qc_dnsmos_ovrl
    dtype: float32
  - name: gt_qc_dnsmos_sig
    dtype: float32
  - name: gt_qc_dnsmos_bak
    dtype: float32
  - name: gt_qc_clip_rate
    dtype: float32
  - name: gt_qc_hf_energy_db
    dtype: float32
  - name: gt_qc_vad_speech_ratio
    dtype: float32
  - name: gt_qc_lead_sil
    dtype: float32
  - name: gt_qc_trail_sil
    dtype: float32
  - name: gt_qc_fails_prompt_qc
    dtype: bool
  - name: simref_qc_dnsmos_ovrl
    dtype: float32
  - name: simref_qc_dnsmos_sig
    dtype: float32
  - name: simref_qc_dnsmos_bak
    dtype: float32
  - name: simref_qc_clip_rate
    dtype: float32
  - name: simref_qc_hf_energy_db
    dtype: float32
  - name: simref_qc_vad_speech_ratio
    dtype: float32
  - name: simref_qc_lead_sil
    dtype: float32
  - name: simref_qc_trail_sil
    dtype: float32
  - name: simref_qc_fails_prompt_qc
    dtype: bool
  - name: qc_hf_energy_db
    dtype: float64
  splits:
  - name: main
    num_bytes: 546214400
    num_examples: 1500
  download_size: 546214400
  dataset_size: 546214400
- config_name: ky
  features:
  - name: utt
    dtype: string
  - name: lang
    dtype: string
  - name: subset
    dtype: string
  - name: source
    dtype: string
  - name: cv_version
    dtype: string
  - name: prompt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: prompt_audio_orig
    dtype: audio
  - name: prompt_text
    dtype: string
  - name: prompt_dur
    dtype: float64
  - name: prompt_sr_orig
    dtype: int64
  - name: prompt_dur_bin
    dtype: string
  - name: prompt_cv_path
    dtype: string
  - name: prompt_split_origin
    dtype: string
  - name: speaker_id
    dtype: string
  - name: speaker_gender
    dtype: string
  - name: speaker_gender_source
    dtype: string
  - name: speaker_age
    dtype: string
  - name: speaker_accent_label
    dtype: string
  - name: speaker_n_in_subset
    dtype: int64
  - name: qc_vad_speech_ratio
    dtype: float64
  - name: qc_lead_sil
    dtype: float64
  - name: qc_trail_sil
    dtype: float64
  - name: qc_clip_rate
    dtype: float64
  - name: qc_bandwidth_hz
    dtype: float64
  - name: qc_spk_centroid_dist
    dtype: float32
  - name: qc_dnsmos_ovrl
    dtype: float32
  - name: qc_dnsmos_sig
    dtype: float32
  - name: qc_dnsmos_bak
    dtype: float32
  - name: text
    dtype: string
  - name: text_norm
    dtype: string
  - name: n_words
    dtype: int64
  - name: n_chars
    dtype: int64
  - name: len_bin
    dtype: string
  - name: phones
    dtype: string
  - name: n_phones
    dtype: int64
  - name: punct_type
    dtype: string
  - name: text_cv_path
    dtype: string
  - name: text_split_origin
    dtype: string
  - name: gt_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_gt
    dtype: bool
  - name: gt_dur
    dtype: float64
  - name: gt_speaker_id
    dtype: string
  - name: gt_same_speaker
    dtype: bool
  - name: gt_gender
    dtype: string
  - name: sim_ref_audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: has_sim_ref
    dtype: bool
  - name: sim_ref_dur
    dtype: float64
  - name: sim_ref_text
    dtype: string
  - name: category
    dtype: string
  - name: subcategory
    dtype: string
  - name: tags
    list: string
  - name: difficulty
    dtype: int64
  - name: has_digit
    dtype: bool
  - name: has_abbrev
    dtype: bool
  - name: has_foreign
    dtype: bool
  - name: notes
    dtype: string
  - name: sim_ref_dur_bin
    dtype: string
  - name: anchor_asr
    dtype: string
  - name: anchor_wer
    dtype: float32
  - name: anchor_cer
    dtype: float32
  - name: anchor_hyp
    dtype: string
  - name: anchor_subs
    dtype: int32
  - name: anchor_dels
    dtype: int32
  - name: anchor_ins
    dtype: int32
  - name: anchor_n_ref_words
    dtype: int32
  - name: anchor_cer_err
    dtype: int32
  - name: anchor_n_ref_chars
    dtype: int32
  - name: anchor_wer_mms
    dtype: float32
  - name: anchor_cer_mms
    dtype: float32
  - name: anchor_subs_mms
    dtype: int32
  - name: anchor_dels_mms
    dtype: int32
  - name: anchor_ins_mms
    dtype: int32
  - name: anchor_cer_err_mms
    dtype: int32
  - name: anchor_wer_scribe
    dtype: float32
  - name: anchor_cer_scribe
    dtype: float32
  - name: anchor_subs_scribe
    dtype: int32
  - name: anchor_dels_scribe
    dtype: int32
  - name: anchor_ins_scribe
    dtype: int32
  - name: anchor_cer_err_scribe
    dtype: int32
  - name: anchor_hyp_scribe
    dtype: string
  - name: anchor_sim_wavlm_sv
    dtype: float32
  - name: anchor_sim_wavlm_ft
    dtype: float32
  - name: anchor_sim_ecapa
    dtype: float32
  - name: gt_qc_dnsmos_ovrl
    dtype: float32
  - name: gt_qc_dnsmos_sig
    dtype: float32
  - name: gt_qc_dnsmos_bak
    dtype: float32
  - name: gt_qc_clip_rate
    dtype: float32
  - name: gt_qc_hf_energy_db
    dtype: float32
  - name: gt_qc_vad_speech_ratio
    dtype: float32
  - name: gt_qc_lead_sil
    dtype: float32
  - name: gt_qc_trail_sil
    dtype: float32
  - name: gt_qc_fails_prompt_qc
    dtype: bool
  - name: simref_qc_dnsmos_ovrl
    dtype: float32
  - name: simref_qc_dnsmos_sig
    dtype: float32
  - name: simref_qc_dnsmos_bak
    dtype: float32
  - name: simref_qc_clip_rate
    dtype: float32
  - name: simref_qc_hf_energy_db
    dtype: float32
  - name: simref_qc_vad_speech_ratio
    dtype: float32
  - name: simref_qc_lead_sil
    dtype: float32
  - name: simref_qc_trail_sil
    dtype: float32
  - name: simref_qc_fails_prompt_qc
    dtype: bool
  - name: qc_hf_energy_db
    dtype: float64
  splits:
  - name: main
    num_bytes: 289269765
    num_examples: 700
  download_size: 289269765
  dataset_size: 289269765
configs:
- config_name: en-US
  data_files:
  - split: main
    path: en-US/main-*
- config_name: es-ES
  data_files:
  - split: main
    path: es-ES/main-*
- config_name: es-MX
  data_files:
  - split: main
    path: es-MX/main-*
- config_name: nl-NL
  data_files:
  - split: main
    path: nl-NL/main-*
- config_name: pt-BR
  data_files:
  - split: main
    path: pt-BR/main-*
- config_name: ky
  data_files:
  - split: main
    path: ky/main-*
- config_name: metrics-wer
  data_files:
  - split: train
    path: reports/per_utterance/wer_*.parquet
- config_name: metrics-sim
  data_files:
  - split: train
    path: reports/per_utterance/sim_*_anchor_*.parquet
- config_name: metrics-sim-floor
  data_files:
  - split: train
    path: reports/per_utterance/sim_*_floor_*.parquet
- config_name: metrics-quality
  data_files:
  - split: train
    path: reports/per_utterance/quality_*.parquet
license: cc0-1.0
task_categories:
- text-to-speech
- automatic-speech-recognition
language:
- en
- es
- pt
- nl
- ky
multilinguality: multilingual
source_datasets:
- fsicoli/common_voice_17_0
annotations_creators:
- crowdsourced
- machine-generated
tags:
- tts
- voice-cloning
- zero-shot-tts
- speech-synthesis
- benchmark
- evaluation
- common-voice
- seed-tts-eval
- speaker-similarity
- intelligibility
pretty_name: Multilingual Speech Benchmark for Zero-Shot TTS
size_categories:
- 1K<n<10K
---

# Multilingual Speech Benchmark for Zero-Shot TTS

A voice-cloning and intelligibility benchmark for six language variants, built
from Common Voice 17.0 by coverage-driven selection rather than random sampling.
Every example pairs a **reference clip of one speaker** with a **target text that
speaker never read**, so a system is asked to clone a voice and produce new
speech, which is what zero-shot TTS is actually for.

**Pipeline source code:** [https://github.com/nineninesix-ai/make-speech-benchmark](https://github.com/nineninesix-ai/make-speech-benchmark) — every number in this card is
reproducible from it. Card generated at pipeline revision `cfe006d`.

**Version 2.0.** The audio, the row composition and the `utt` identifiers are
unchanged from v1: anything already synthesised against v1 still joins. What
changed is the measurement and the description of it — see
[What changed in v2.0](#what-changed-in-v20).

---

## Contents

- [What this measures, and what it does not](#what-this-measures-and-what-it-does-not)
- [Quick start](#quick-start)
- [What changed in v2.0](#what-changed-in-v20)
- [Subsets](#subsets)
- [Human anchors — intelligibility](#human-anchors--intelligibility)
- [Human anchors — speaker similarity](#human-anchors--speaker-similarity)
- [Can a model beat the anchor?](#can-a-model-beat-the-anchor)
- [Evaluation protocol](#evaluation-protocol)
- [How to read the numbers](#how-to-read-the-numbers)
- [Source data](#source-data)
- [Quality control](#quality-control)
- [Example selection](#example-selection)
- [Data fields](#data-fields)
- [Reports](#reports)
- [Limitations](#limitations)
- [Reproduction](#reproduction)
- [Personal and sensitive information](#personal-and-sensitive-information)
- [Licence, citation, contact](#licence-citation-contact)

---

## What this measures, and what it does not

This is an **intelligibility and voice-cloning benchmark**, not a TTS quality
benchmark. It measures two things:

- **WER / CER** on a recogniser reading the synthesis — a proxy for intelligibility;
- **SIM** — cosine similarity of speaker embeddings between the synthesis and the
  reference clip — a proxy for speaker identity.

It measures **no naturalness axis at all**: no subjective MOS or CMOS, and no
predicted naturalness on the synthesis. A system can score 3 % WER and 0.95 SIM
and still sound robotic, and nothing here would notice. It also contains **no
digits, no abbreviations and no long-form text** — Common Voice sentences of 3-16
words — so text normalisation, the most common production TTS failure, is
untested. Treat a good score as a necessary condition, not a sufficient one.

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("nineninesix/multilingual-speech-benchmark", "en-US", split="main")
row = ds[0]
row["prompt_audio"]   # reference clip of the speaker, 16 kHz
row["text"]           # the target text, which this speaker never read
row["gt_audio"]       # a real recording of that text, by someone else
row["sim_ref_audio"]  # a second clip of the PROMPT speaker -> the SIM anchor

row["anchor_wer"]           # what the recogniser scores on the HUMAN recording
row["anchor_sim_wavlm_ft"]  # what two recordings of the same person score
```

Those last two are the reference points your model's numbers are read against,
and they ship per row — along with the edit counts, so the corpus-level headline
is derivable without leaving the dataset (see [Data fields](#data-fields)). The measurements behind them — S/D/I counts, transcripts,
every recogniser and encoder — are browsable as separate configs:

```python
wer = load_dataset("nineninesix/multilingual-speech-benchmark", "metrics-wer", split="train")
sim = load_dataset("nineninesix/multilingual-speech-benchmark", "metrics-sim", split="train")
```

Synthesise `row["text"]` conditioned on `row["prompt_audio"]`, save it as
`{utt}.wav`, then score:

```bash
git clone https://github.com/nineninesix-ai/make-speech-benchmark && cd make-speech-benchmark
uv venv && source .venv/bin/activate && uv pip install -e ".[all]"

msbench-wer --lang en-US --audio synth --asr whisper \
    --synth-dir out/mymodel/en-US --model-name mymodel
msbench-sim --lang en-US --mode synth --encoder wavlm_ft \
    --synth-dir out/mymodel/en-US --model-name mymodel
msbench-report --all
```

### seed-tts-eval compatibility

The layout is consumable by the unmodified seed-tts-eval scripts, but read
[Human anchors — speaker similarity](#human-anchors--speaker-similarity) first:
those scripts score SIM with `wavlm_large_finetune.pth`, whose values live near
0.65-0.69 here, while v1 published anchors from `microsoft/wavlm-base-plus-sv`,
which live near 0.93. Comparing one against the other is the single easiest way
to misread this benchmark, and v2 ships anchors for both.

## What changed in v2.0

**No data changed.** Audio in all three streams is byte-identical, rows are in
the same order, `utt` is stable. Verified by `msbench.build.validate`, which is run
before every release.

### Published numbers

The headline WER changes because the **aggregation** changed, not because the
recogniser or the audio did. v1 averaged per-utterance error rates; v2 reports
the corpus-level rate that seed-tts-eval uses, Σ(S+D+I) / Σ N_ref, and keeps the
old form beside it labelled as legacy. On texts with a median of 6-7 words the
two differ materially: one wrong word in a four-word sentence contributes 25 %
under the macro mean regardless of its weight in the corpus.

Each column below isolates **one** change, measured on identical audio:

| lang   |   published v1 (macro) |   legacy (macro) |   drift (env) |   D-02 penalty |   D-10 chunking |   D-01 aggregation |   v2 (corpus) | v2 corpus 95% CI   |
|:-------|-----------------------:|-----------------:|--------------:|---------------:|----------------:|-------------------:|--------------:|:-------------------|
| en-US  |                 0.0806 |           0.0805 |       -0.0001 |         0.0002 |               0 |            -0.0033 |        0.0774 | [0.0703, 0.0842]   |
| es-ES  |                 0.0485 |           0.0485 |        0      |        -0.0001 |               0 |            -0.0032 |        0.0452 | [0.0397, 0.0512]   |
| es-MX  |                 0.0654 |           0.0652 |       -0.0002 |        -0.0004 |               0 |            -0.0035 |        0.0613 | [0.0541, 0.0691]   |
| nl-NL  |                 0.0407 |           0.0407 |        0      |         0.0001 |               0 |            -0.0023 |        0.0385 | [0.0331, 0.0453]   |
| pt-BR  |                 0.0823 |           0.0816 |       -0.0007 |         0.0001 |               0 |            -0.008  |        0.0737 | [0.0608, 0.0884]   |
| ky     |                 0.1104 |           0.1104 |        0      |         0      |               0 |            -0.0106 |        0.0998 | [0.0773, 0.1311]   |

- **drift (env)** — replaying v1's exact decoding parameters on current library
  and hardware versions. Worst case −0.0007; four subsets reproduce exactly. This
  is not a methodological change and is reported only so it is not mistaken for one.
- **D-02 penalty** — v1 decoded with `repetition_penalty=1.1`, which is not
  standard for WER evaluation and suppresses exactly the token pattern that
  autoregressive TTS failure produces. Removing it moves the **human** anchor by
  at most 0.0004, which is expected: human speech rarely loops. Whether it was
  masking model failures is a claim about synthesis and is **not** tested here.
- **D-10 chunking** — `chunk_length_s=30` engaged Whisper's long-form path on
  clips of 2-6 s. Effect: exactly zero, transcripts 100 % identical. Removed as
  hygiene.
- **D-01 aggregation** — the entire remaining change.

### Schema

| change | detail |
|---|---|
| `qc_hf_energy_db` | new name for `qc_bandwidth_hz`, which holds **decibels**, not hertz: it is the share of energy above 5 kHz in dB. The old column is kept as a deprecated copy for one release |
| `tags` | was `list<null>`, a type that cannot hold a value; now `list<string>` |
| `sim_ref_dur_bin` | duration stratum of the *second* clip, so SIM breakdowns can control for it |
| `gt_qc_*`, `simref_qc_*` | 9 columns each: DNSMOS and the QC battery on ground-truth and second-clip audio, which never faced the prompt-acceptance filters. **Reporting only — nothing is filtered** |
| `phones` (ky) | espeak dental markers `t[` / `d[` (1,584 and 1,132 occurrences) replaced by IPA `t̪` / `d̪`. `n_phones` is unaffected |
| removed | `qc_snr_db` and `qc_asr_cer` were declared in v1 and **100 % null in every subset** — never computed |

### New measurements

- a second recogniser on all six subsets (`facebook/mms-1b-all`, CTC without a
  language model) and a third encoder family for SIM;
- an **impostor floor** for SIM, which v1 lacks entirely;
- **95 % confidence intervals** on every headline number, and `n` in every cell;
- **per-utterance parquet artifacts** shipped inside this repository, so any
  number here can be recomputed, sliced or re-tested without a GPU.

### Corrections to the v1 card

| claim in v1 | corrected |
|---|---|
| second clip missing for "26-39 %" of rows | present for **60.6-87.9 %**; see [Subsets](#subsets) |
| SIM rises monotonically with prompt duration in five subsets | **three** under v1's encoder — en-US, es-MX, nl-NL |
| every example carries ground-truth audio and a second clip | GT missing for 7 examples; second clip present for 60.6-87.9 % |
| "Speakers" | **speakers used**, which is not speakers available in the source slice |
| anchors are a "physical ceiling" | **human anchor**. A model can exceed it — see [Can a model beat the anchor?](#can-a-model-beat-the-anchor) |

## Subsets

| subset   |   examples |   speakers used | with GT audio   | with second clip   |
|:---------|-----------:|----------------:|:----------------|:-------------------|
| en-US    |       1500 |            1162 | 1500/1500       | 909/1500           |
| es-ES    |       1500 |             722 | 1499/1500       | 1222/1500          |
| es-MX    |       1500 |             929 | 1500/1500       | 1201/1500          |
| nl-NL    |       1500 |             469 | 1500/1500       | 1319/1500          |
| pt-BR    |       1500 |             247 | 1500/1500       | 983/1500           |
| ky       |        700 |             181 | 694/700         | 599/700            |

## Human anchors — intelligibility

Every figure below is a **human anchor**: a recogniser reading a real human
recording of the target text. Without it a model's WER cannot be read — is 8 %
on en-US a weak model, or roughly what the recogniser scores on human speech?

Headline aggregation is corpus-level, Σ(S+D+I) / Σ N_ref. Intervals are 95 %
cluster bootstraps resampling **speakers**, not rows.

| lang   | ASR     |    n |   speakers |   WER corpus | 95% CI           |   WER macro (v1) |   CER corpus | exact   | catastrophic   |
|:-------|:--------|-----:|-----------:|-------------:|:-----------------|-----------------:|-------------:|:--------|:---------------|
| en-US  | whisper | 1500 |       1012 |       0.0774 | [0.0703, 0.0842] |           0.0807 |       0.0296 | 59.7%   | 1.4%           |
| es-ES  | whisper | 1499 |        574 |       0.0452 | [0.0397, 0.0512] |           0.0484 |       0.0154 | 73.0%   | 0.9%           |
| es-MX  | whisper | 1500 |        639 |       0.0613 | [0.0541, 0.0691] |           0.0648 |       0.0214 | 68.3%   | 1.3%           |
| nl-NL  | whisper | 1500 |        236 |       0.0385 | [0.0331, 0.0453] |           0.0408 |       0.0108 | 76.0%   | 0.5%           |
| pt-BR  | whisper | 1500 |        229 |       0.0737 | [0.0608, 0.0884] |           0.0817 |       0.0234 | 71.5%   | 3.5%           |
| ky     | gigaam  |  694 |         89 |       0.0998 | [0.0773, 0.1311] |           0.1104 |       0.0293 | 66.4%   | 5.5%           |

`exact` is the share of utterances transcribed with zero errors.
`catastrophic` is the share above 50 % WER — the indicator that catches looping,
babbling and dropped clauses long before they move the mean.

### Three recognisers, and why the gaps matter

Whisper decodes with a strong internal language model, so a mispronounced or
half-swallowed word is often repaired into the word the sentence implies; a
system is then credited with intelligibility it did not produce. MMS is CTC with
greedy decoding and no language model: it emits what it heard. Its absolute WER
is higher everywhere, which is not a defect. **Scribe** is a second strong-LM
read, from a different vendor and training set.

| lang   | A          | B       |   WER A |   WER B |   delta (A−B) | 95% CI             | identical transcripts   |   n shared |
|:-------|:-----------|:--------|--------:|--------:|--------------:|:-------------------|:------------------------|-----------:|
| en-US  | elevenlabs | whisper |  0.0515 |  0.0774 |       -0.0259 | [-0.0309, -0.0209] | 68.2%                   |       1500 |
| en-US  | elevenlabs | mms     |  0.0515 |  0.1784 |       -0.1269 | [-0.1357, -0.1183] | 32.1%                   |       1500 |
| en-US  | whisper    | mms     |  0.0774 |  0.1784 |       -0.1011 | [-0.1086, -0.0934] | 32.0%                   |       1500 |
| es-ES  | elevenlabs | whisper |  0.0269 |  0.0452 |       -0.0183 | [-0.0227, -0.0143] | 78.8%                   |       1499 |
| es-ES  | elevenlabs | mms     |  0.0269 |  0.1248 |       -0.0979 | [-0.1065, -0.0895] | 49.0%                   |       1499 |
| es-ES  | whisper    | mms     |  0.0452 |  0.1248 |       -0.0796 | [-0.0871, -0.0717] | 50.0%                   |       1499 |
| es-MX  | elevenlabs | whisper |  0.0351 |  0.0613 |       -0.0262 | [-0.0321, -0.0207] | 73.9%                   |       1500 |
| es-MX  | elevenlabs | mms     |  0.0351 |  0.1482 |       -0.1131 | [-0.1240, -0.1028] | 42.0%                   |       1500 |
| es-MX  | whisper    | mms     |  0.0613 |  0.1482 |       -0.0869 | [-0.0952, -0.0788] | 43.0%                   |       1500 |
| nl-NL  | elevenlabs | whisper |  0.0159 |  0.0385 |       -0.0226 | [-0.0277, -0.0186] | 78.9%                   |       1500 |
| nl-NL  | elevenlabs | mms     |  0.0159 |  0.078  |       -0.0621 | [-0.0730, -0.0540] | 58.7%                   |       1500 |
| nl-NL  | whisper    | mms     |  0.0385 |  0.078  |       -0.0395 | [-0.0478, -0.0328] | 57.3%                   |       1500 |
| pt-BR  | elevenlabs | whisper |  0.0531 |  0.0737 |       -0.0207 | [-0.0307, -0.0120] | 73.7%                   |       1500 |
| pt-BR  | elevenlabs | mms     |  0.0531 |  0.2242 |       -0.1712 | [-0.1953, -0.1497] | 36.2%                   |       1500 |
| pt-BR  | whisper    | mms     |  0.0737 |  0.2242 |       -0.1505 | [-0.1682, -0.1333] | 35.4%                   |       1500 |
| ky     | elevenlabs | mms     |  0.1829 |  0.2425 |       -0.0596 | [-0.0838, -0.0361] | 20.3%                   |        694 |
| ky     | elevenlabs | gigaam  |  0.1829 |  0.0998 |        0.0831 | [0.0629, 0.1046]   | 43.1%                   |        694 |
| ky     | mms        | gigaam  |  0.2425 |  0.0998 |        0.1427 | [0.1273, 0.1593]   | 24.8%                   |        694 |

Deltas are paired bootstraps over the utterances both recognisers scored; every
interval above excludes zero.

Two of these gaps mean different things.

**Whisper − MMS** measures how much the intelligible reading depends on the
listener's expectations. A large gap means the acoustics alone do not carry the
sentence.

**Scribe − Whisper** measures something the benchmark could not see with one
recogniser: how much of the "human anchor" was never the human at all. Scribe
reads the *same recordings* 27-59 % more accurately, and with a lower
catastrophic rate, so it is not buying accuracy with hallucination. On `nl-NL`
the human anchor falls from 0.0385 to **0.0159** with **zero** catastrophic
utterances. Whatever a v1-style anchor attributed to "human speech is hard" was
substantially Whisper's own error, and the ceiling a synthesis system is measured
against is correspondingly higher.

`ky` is the exception and runs the other way: Scribe (0.1829) is well behind
GigaAM (0.0998), paired delta +0.0831 [0.0629, 0.1046]. See below.

### Is the Kyrgyz subset usable?

Kyrgyz is the one subset where the primary recogniser (`GigaAM-Multilingual`) is
not independently validated for this purpose, and its v1 anchor of 11.04 % sat
above the 10 % threshold that calls for a cross-check. v2 performs that
cross-check with MMS, which covers Kyrgyz through its `kir` adapter.

MMS scores 2.03x-3.04x the strong recogniser across the five Whisper subsets.
Kyrgyz sits at **2.43x** — mid-range. GigaAM therefore behaves, relative to a
language-model-free CTC baseline, exactly as Whisper does on the other five
subsets, and the cross-check does **not** find it anomalous.

A third recogniser now confirms it from the other direction. Scribe covers `kir`
— which Whisper's 100 languages do not — and it lands at 0.1829, well *behind*
GigaAM's 0.0998 (paired delta +0.0831 [0.0629, 0.1046]) while beating MMS. So on
the five subsets where a general-purpose hosted model is the most accurate reader
available, it is; on Kyrgyz, the language-specific model wins by a wide margin.
That is what a genuinely competent `ky` recogniser looks like, and it is the
opposite of what a broken one would produce.

Scribe's Kyrgyz output also carries a signature worth knowing about. The
references use exactly the 36 letters of the Kyrgyz alphabet; **6.2 %** of Scribe
transcripts contain letters from neighbouring Turkic languages — `ғ`, `қ`, `ұ`,
`ә`, `і` (Kazakh), `ҡ` (Bashkir). Folding those to their Kyrgyz counterparts
recovers only 0.0063 WER, 3.4 % of its errors, so the spelling is not what makes
it worse. It is a **symptom**: rows containing a foreign letter average 0.59 WER
against 0.17 elsewhere. When Scribe drifts into a neighbouring orthography it is
usually losing the whole utterance, not just the spelling.

The real limitation of `ky` is a sampling problem: 694 rows come from only
**89 distinct ground-truth speakers**, and the anchor's interval is
correspondingly wide, [0.0773, 0.1311]. Use it for measurement, treat small
differences between systems on it with suspicion, and quote the interval.

## Human anchors — speaker similarity

**An absolute SIM value is meaningless without a floor.** v1 published an anchor
and no baseline, so "0.87 against a ceiling of 0.93" read like 94 % of the way
there. Whether that is good depends entirely on what two *different* speakers
score, and v1 never measured it. v2 does, over 5,000 random cross-speaker prompt
pairs per language per encoder:

| lang   | encoder   |   anchor | 95% CI           |   floor mean |   floor p95 |   usable range |   n rows |   speakers |   distinct values |
|:-------|:----------|---------:|:-----------------|-------------:|------------:|---------------:|---------:|-----------:|------------------:|
| en-US  | wavlm_sv  |   0.9317 | [0.9283, 0.9347] |       0.6119 |      0.8707 |         0.3198 |      909 |        631 |               629 |
| en-US  | wavlm_ft  |   0.652  | [0.6424, 0.6601] |       0.0725 |      0.2483 |         0.5795 |      909 |        631 |               630 |
| en-US  | ecapa     |   0.6141 | [0.6042, 0.6227] |       0.0822 |      0.2447 |         0.5319 |      909 |        631 |               631 |
| es-ES  | wavlm_sv  |   0.9435 | [0.9392, 0.9469] |       0.7286 |      0.9245 |         0.2149 |     1222 |        590 |               587 |
| es-ES  | wavlm_ft  |   0.6888 | [0.6791, 0.6980] |       0.1302 |      0.3183 |         0.5586 |     1222 |        590 |               590 |
| es-ES  | ecapa     |   0.6505 | [0.6405, 0.6599] |       0.132  |      0.3206 |         0.5185 |     1222 |        590 |               589 |
| es-MX  | wavlm_sv  |   0.9456 | [0.9432, 0.9480] |       0.7315 |      0.932  |         0.2141 |     1201 |        771 |               768 |
| es-MX  | wavlm_ft  |   0.6882 | [0.6811, 0.6956] |       0.1685 |      0.3576 |         0.5197 |     1201 |        771 |               769 |
| es-MX  | ecapa     |   0.635  | [0.6268, 0.6438] |       0.122  |      0.299  |         0.5131 |     1201 |        771 |               769 |
| nl-NL  | wavlm_sv  |   0.9285 | [0.9235, 0.9331] |       0.7446 |      0.9215 |         0.184  |     1319 |        412 |               412 |
| nl-NL  | wavlm_ft  |   0.6708 | [0.6619, 0.6797] |       0.174  |      0.352  |         0.4969 |     1319 |        412 |               412 |
| nl-NL  | ecapa     |   0.6232 | [0.6134, 0.6336] |       0.1866 |      0.3666 |         0.4366 |     1319 |        412 |               412 |
| pt-BR  | wavlm_sv  |   0.9339 | [0.9265, 0.9403] |       0.765  |      0.9248 |         0.1689 |      983 |        161 |               161 |
| pt-BR  | wavlm_ft  |   0.6346 | [0.6166, 0.6502] |       0.1864 |      0.3754 |         0.4482 |      983 |        161 |               161 |
| pt-BR  | ecapa     |   0.5763 | [0.5578, 0.5930] |       0.1257 |      0.2993 |         0.4506 |      983 |        161 |               161 |
| ky     | wavlm_sv  |   0.9214 | [0.9121, 0.9302] |       0.7173 |      0.9279 |         0.204  |      599 |        155 |               155 |
| ky     | wavlm_ft  |   0.6069 | [0.5849, 0.6267] |       0.184  |      0.3837 |         0.4229 |      599 |        155 |               155 |
| ky     | ecapa     |   0.5593 | [0.5371, 0.5811] |       0.1484 |      0.3428 |         0.411  |      599 |        155 |               155 |

- **anchor** — cos(second clip, prompt clip): two different recordings of the
  same person.
- **floor mean / p95** — cross-speaker pairs. p95 is the practical false-accept
  level: a system scoring there is being confused with strangers one time in twenty.
- **usable range** — anchor minus floor. This is the entire span in which a
  cloning system can distinguish itself.

**Read the `wavlm_sv` rows carefully.** That is v1's encoder, and its usable
range is 0.17-0.32 on a scale that looks like it runs to 1.0. Worse, its
impostor **p95** reaches 0.87-0.93 — on Kyrgyz the p95 impostor (0.9279) is
*above* the human anchor (0.9214), meaning more than 5 % of random cross-speaker
pairs outscore two recordings of the same person. A bare number near 0.93 from
this encoder carries very little information. `wavlm_ft` and `ecapa` keep
0.41-0.58 of usable range and separate speakers far more cleanly.

Report the normalised score, which is readable where a bare cosine is not:

```
sim_norm = (sim_o − sim_floor) / (sim_anchor − sim_floor)
```

0 means indistinguishable from an impostor; 1 means as close as two recordings
of the same person. Per-language coefficients are in `reports/csv/sim.csv`.

### Which encoder to use

| encoder | model | use it for |
|---|---|---|
| `wavlm_sv` | `microsoft/wavlm-base-plus-sv` | continuity with v1 numbers only |
| `wavlm_ft` | `wavlm_large_finetune.pth` | comparability with the seed-tts-eval literature |
| `ecapa` | `speechbrain/spkrec-ecapa-voxceleb` | a reading from outside the WavLM family |

The third exists because of a circularity in the build: prompt QC rejected clips
lying more than μ−2σ from their speaker's centroid **as measured by WavLM-SV**,
and the anchor was then measured with WavLM-SV. The pool was pre-selected for
homogeneity in the metric's own space. `wavlm_ft` shares that backbone and
inherits the blind spot; ECAPA-TDNN shares neither architecture, pretraining nor
training corpus, so where its anchor tracks WavLM's the anchor is a property of
the speakers, and where it does not, the selection is showing through.

### SIM against prompt duration

Under v1's encoder, the relationship is monotonic in **three** subsets, not five:

| lang   |   <3.7 |   3.7-4.5 |   >=4.5 | monotonic   |
|:-------|-------:|----------:|--------:|:------------|
| en-US  | 0.9228 |    0.9284 |  0.9367 | True        |
| es-ES  | 0.9321 |    0.9476 |  0.9456 | False       |
| es-MX  | 0.9377 |    0.9455 |  0.9492 | True        |
| ky     | 0.9167 |    0.9321 |  0.9138 | False       |
| nl-NL  | 0.925  |    0.9309 |  0.9367 | True        |
| pt-BR  | 0.9262 |    0.9491 |  0.9399 | False       |

The choice of encoder changes the answer: under `wavlm_ft` and `ecapa`, es-ES
becomes monotonic too and only `ky` and `pt-BR` peak in the middle bin. Full
tables with `n` per cell are in `reports/csv/sim_breakdown.csv`.

### The confounder v1 did not control

The table above bins on the **prompt** clip only. The other clip in the pair —
`sim_ref_audio`, down to about 2 s — was uncontrolled, and it moves the anchor:

| lang   |   3.5-4.5 |   <3.5 |   >=4.5 |
|:-------|----------:|-------:|--------:|
| en-US  |    0.948  | 0.9268 |  0.9432 |
| es-ES  |    0.9528 | 0.9413 |  0.9572 |
| es-MX  |    0.9519 | 0.9432 |  0.9574 |
| ky     |    0.911  | 0.9223 |  0.9174 |
| nl-NL  |    0.9431 | 0.9279 |  0.8895 |
| pt-BR  |    0.9451 | 0.9327 |  0.9376 |

`sim_ref_dur_bin` ships as a column in v2 so any SIM breakdown can control for it.

### Effective sample size

The anchor is **constant within a speaker** by construction: every example of a
speaker shares one prompt clip and one second clip. So `n rows` is not the number
of measurements — en-US's 909 rows carry 629 distinct values, and `ky`'s 599 rows
carry 155. Every interval in this card is a cluster bootstrap over speakers for
this reason. A row-level bootstrap would report intervals roughly √(rows per
speaker) too narrow.

## Can a model beat the anchor?

**Yes, and models do.** The Seed-TTS paper's Table 1 (zero-shot in-context
learning) reports English SIM 0.762 against a human 0.730, and Mandarin WER
1.115 % against a human 1.254 %. The anchor is not a physical ceiling and this
card does not call it one.

The reason is structural. Synthesis is **conditioned on the prompt** and inherits
its recording channel, microphone and room; the anchor compares two *different*
recordings of the person, made at different times. The comparison is asymmetric
in the model's favour. Three further biases point in various directions:

- **circularity** — the QC filter and the v1 SIM metric share an embedding space
  (above);
- **transcript quality** — the WER anchor mixes recogniser error, reader error
  and unverified crowd-sourced transcripts. The `ky` subset contains clear cases
  where both recognisers agree with each other and disagree with the reference,
  i.e. the reference is wrong;
- **training contamination** — Whisper has plausibly seen Common Voice in
  training, which pushes the anchor the other way.

None of these are quantified. Treat the anchor as a reference point, not a bound.

## Evaluation protocol

Every parameter that determines a number. v1 stated none of them.

### Preprocessing parity

The anchors are measured on audio that the build put into a canonical state:
mono, 16 kHz, Silero-VAD trimmed with a 50 ms pad, peak-normalised to −1 dBFS.
**Your synthesis must go through the same pipeline** or the comparison is not
like-for-like — trailing silence and level differences both shift SIM.
`msbench.audio.prepare()` does this and the drivers apply it by default.

Resampling uses **soxr HQ**. This matters: v1 resampled with `np.interp`, linear
interpolation with no anti-aliasing filter. References are natively 16 kHz and
never took that path, but synthesis at 22.05 / 24 / 44.1 kHz always did and
arrived at the speaker encoder carrying aliasing artifacts — a 15 kHz tone that
must vanish at 16 kHz survives at −3.1 dBr with 93 % of the residue folded onto
1 kHz, in the middle of the speech band. Every model's SIM was depressed by this;
the anchor was not.

Stored dataset audio is **not** re-trimmed, because re-running VAD with a
different Silero version than the build used would move the anchor.

### ASR

```
model            openai/whisper-large-v3        (ai-sage/GigaAM-Multilingual rev "ctc" for ky)
dtype            float16
num_beams        1                              greedy
max_new_tokens   200                            the runaway guard
repetition_penalty  not set                     removed in v2
chunk_length_s   not set                        short-form path
batch            8
second opinion   facebook/mms-1b-all, per-language adapter, CTC greedy, no LM
```

### ElevenLabs Scribe

```
model               scribe_v2
language_code       per subset, ISO 639-3 (eng / spa / nld / por / kir) — never auto-detect
tag_audio_events    false     default is TRUE; its tags would be scored as words
diarize             false
timestamps_granularity  none
seed                0
temperature         0
no_verbatim         false     it strips filler words, which are real content here
keyterms            unset     it biases decoding toward supplied words
workers             12        measured: ~8.8 clips/s; 24 draws 429s
timeout             300 s
```

Two properties of this backend have no counterpart in the local recognisers and
must be understood before its numbers are used.

**It is not deterministic, even pinned.** Identical requests return different
transcripts. `seed` alone changes nothing; `seed` with `temperature=0` narrows
the spread a great deal but does not close it. Measured by running `ky` three
times end to end: only **69.5 %** of transcripts are identical across all three,
yet corpus WER lands at 0.1829 / 0.1805 / 0.1809 — a range of **0.0024** against
a bootstrap interval 0.078 wide. The noise is real per utterance and negligible
in the aggregate. Audit a single row and you may not reproduce it; quote a
headline and you will. (`scribe_v1` is less stable still, which is why v2 is the
default here — the version number is not the reason.)

**It hallucinates on short clips.** On a 1.1 s pt-BR utterance where Whisper
scores 0.00 it returned an unrelated sentence. `audio_duration_secs` comes back
matching the clip, so this is a decoder failure, not a transport one. Read its
`catastrophic_rate` alongside its WER, never the WER alone.

There is no batch endpoint: the API takes one file per request, and "batch" in
ElevenLabs' terms means asynchronous delivery by webhook, which does not raise
throughput. Concurrency is the only lever.

### Text normalisation

Applied identically to reference and hypothesis:

```
NFC -> lowercase -> expand digits (num2words, per language; none for ky)
    -> strip punctuation -> collapse whitespace
PUNCT = !"#$%&()*+,-./:;<=>?@[\]^_`{|}~«»„""''…–—¿¡
```

Apostrophes are **kept** (word-internal: don't, 's-Gravenhage). Diacritics are
**kept** — `año`/`ano` and `sé`/`se` are different words, and stripping them
would mask real errors. This is where the normaliser deliberately differs from
`whisper.normalizers`, which is too aggressive for non-English.

### Aggregation

```
wer_corpus  = Σ(S+D+I) / Σ N_ref     headline, matches seed-tts-eval
wer_macro   = mean per-utterance     v1 legacy, reported alongside
cer         counts the space as a character (jiwer default)
exact_match = share of utterances with WER == 0
catastrophic_rate = share with WER > 0.5
```

Rows whose normalised reference is empty are **counted** (`n_empty_ref`), not
silently dropped. In this dataset that count is 0 in every subset — the defect
existed in v1's code but never affected a published number.

## How to read the numbers

**Confidence intervals.** Every headline figure carries a 95 % cluster bootstrap
over speakers (2,000 replicates, seed 20260725). Observations are not
independent: one prompt serves up to 7 examples, GT-speaker concentration is
uncapped, and the SIM anchor is constant within a speaker. Resampling rows would
understate the interval.

**Claiming "A beats B".** Do not compare two independent intervals — that throws
away the fact that both systems were measured on the same sentences. Use
`msbench.stats.paired_bootstrap`, which resamples the shared speakers once per
replicate and applies that resample to both systems, and report the delta with
its interval and P(A better).

**Per-cell `n`.** Every breakdown table in `reports/` carries `n` and the number
of distinct speakers behind it, and flags cells with fewer than 30 speakers.
Do not read a trend off a thin cell.

**Cross-language comparison is not supported.** Anchor-normalised comparison is
valid **within a language only**. Differences between anchors are a property of
the recogniser, not of the languages, and for `ky` it is a different recogniser
entirely.

## Source data

Common Voice 17.0 (`fsicoli/common_voice_17_0`), CC0. Text and audio are
crowd-sourced; speaker metadata is self-declared and often absent.

### Regional variants are not locales

Common Voice has no `es-MX`, `pt-BR` or `nl-NL` locale. These variants live in
free-text, self-declared `accents` / `variant` columns, the field is
multi-valued, and the labels themselves contain commas inside parentheses
("España: Norte peninsular (Asturias, Castilla y León, Cantabria)"). A naive
`split(",")` shreds them. Labels are split on commas at parenthesis depth zero
and matched **exactly** — substring matching gives false positives, since the
Caribbean Spanish label contains "Costa del golfo de México" and would be picked
up by a `contains("México")` test.

`es-ES` means the Castilian norm: Norte peninsular plus Centro-Sur. Andalusian is
excluded — seseo/ceceo places it closer to `es-MX`, and 82 % of that slice is one
speaker. /θ/ is present in `es-ES` and absent from `es-MX`, which is the point of
having both.

## Quality control

Prompt candidates were rejected on criteria that speech restoration cannot fix:
not mono, no speech found, clipping above 1e-3, energy above 5 kHz below −45 dB
(the signature of upsampling from 8 kHz), speech ratio below 0.75 after trimming,
more than one speech segment after merging gaps under 0.25 s, and duration
outside 2-12 s. Noise metrics were deliberately **not** used as filters, because
references are expected to go through a restoration model downstream.

Two things about the `qc_*` columns that v1 did not state:

- they are computed on the **untrimmed** clip, while `prompt_audio` ships
  trimmed. This is why `qc_lead_sil` and `qc_trail_sil` are non-zero on audio
  that has had its edge silence removed;
- they cover **prompts only**.

The last point matters for the WER anchor. `gt_audio` and `sim_ref_audio` got the
same *preprocessing* but never faced the *rejection* filters, so the anchor
includes recordings that would not have been accepted as prompts. v2 measures how
many, and ships the result as `gt_qc_*` and `simref_qc_*` columns:

| lang   | audio   |    n |   DNSMOS OVRL |   OVRL p05 |   clipping % |   narrowband % |   low speech % |   would fail prompt QC % |
|:-------|:--------|-----:|--------------:|-----------:|-------------:|---------------:|---------------:|-------------------------:|
| en-US  | prompt  | 1500 |         2.761 |      1.971 |            0 |           0.27 |           0.27 |                     0.53 |
| en-US  | gt      | 1500 |         2.766 |      1.968 |            0 |           1.47 |           0.93 |                     2.4  |
| en-US  | sim_ref |  909 |         2.752 |      2.028 |            0 |           0.22 |           0    |                     0.22 |
| es-ES  | prompt  | 1500 |         2.766 |      2.011 |            0 |           1.33 |           0.2  |                     1.53 |
| es-ES  | gt      | 1499 |         2.8   |      2.041 |            0 |           2.33 |           0.4  |                     2.74 |
| es-ES  | sim_ref | 1222 |         2.759 |      1.991 |            0 |           1.64 |           0.16 |                     1.8  |
| es-MX  | prompt  | 1500 |         2.68  |      1.887 |            0 |           0.53 |           0    |                     0.53 |
| es-MX  | gt      | 1500 |         2.662 |      1.834 |            0 |           2.73 |           1.8  |                     4.27 |
| es-MX  | sim_ref | 1201 |         2.672 |      1.861 |            0 |           1.25 |           0    |                     1.25 |
| nl-NL  | prompt  | 1500 |         2.808 |      2.152 |            0 |           0.47 |           0    |                     0.47 |
| nl-NL  | gt      | 1500 |         2.871 |      2.218 |            0 |           1.13 |           0.13 |                     1.27 |
| nl-NL  | sim_ref | 1319 |         2.804 |      2.082 |            0 |           1.59 |           0    |                     1.59 |
| pt-BR  | prompt  | 1500 |         2.726 |      2.004 |            0 |           0.4  |           0    |                     0.4  |
| pt-BR  | gt      | 1500 |         2.783 |      2.005 |            0 |           2.47 |           1.07 |                     3.53 |
| pt-BR  | sim_ref |  983 |         2.697 |      2.049 |            0 |           0.61 |           0.61 |                     1.22 |
| ky     | prompt  |  700 |         2.718 |      1.859 |            0 |           1.71 |           0    |                     1.71 |
| ky     | gt      |  694 |         2.749 |      1.924 |            0 |           1.87 |           0.58 |                     2.31 |
| ky     | sim_ref |  599 |         2.671 |      1.812 |            0 |           0.67 |           0    |                     0.67 |

`would fail prompt QC %` applies the prompt thresholds to each stream. Prompts
pass by construction. Nothing is filtered — removing rows would break `utt`
stability with v1 — so a user who wants a clean subset applies their own
threshold and says so.

## Example selection

**Prompts and targets are decoupled.** Each speaker contributes one fixed
reference clip for all of their examples, which keeps SIM variance down; the
target text comes from a different recording, usually a different speaker. The
model is therefore always asked for speech that does not exist.

Selection is constrained rather than random: a target length distribution per
language, a per-speaker cap, a minimum female-voice share where the source allows
one, and full phonetic coverage. All six subsets cover **100 % of the candidate
pool's phoneme and diphone inventory**.

## Data fields

Three audio streams per example, all mono 16 kHz, VAD-trimmed, peak-normalised
to −1 dBFS:

| field | what it is |
|---|---|
| `prompt_audio` | the reference clip — condition your model on this |
| `prompt_audio_orig` | the same clip at its original sample rate |
| `gt_audio` | a real recording of `text`, usually by a **different** speaker. Basis of the WER anchor |
| `sim_ref_audio` | a second clip of the **prompt** speaker, different text. Basis of the SIM anchor |

**The human anchors, per row** — the benchmark's central numbers, carried in the
data itself so they are visible without downloading anything else:

| field | what it is |
|---|---|
| `anchor_asr` | which recogniser produced the primary anchor: `whisper`, or `gigaam` for `ky` |
| `anchor_wer`, `anchor_cer` | that recogniser's **per-utterance** error rate on `gt_audio` |
| `anchor_subs`, `anchor_dels`, `anchor_ins`, `anchor_n_ref_words` | the edit counts behind it, and the reference length |
| `anchor_cer_err`, `anchor_n_ref_chars` | the same for characters |
| `anchor_hyp` | what it actually transcribed, normalised — so a row's WER can be understood rather than only read |
| `anchor_wer_mms`, `anchor_cer_mms` + counts | MMS, CTC without a language model |
| `anchor_wer_scribe`, `anchor_cer_scribe` + counts | ElevenLabs Scribe v2, and `anchor_hyp_scribe` |
| `anchor_sim_wavlm_sv`, `anchor_sim_wavlm_ft`, `anchor_sim_ecapa` | cos(second clip, prompt clip) under each encoder |

`NaN` where the underlying audio is absent (`has_gt` or `has_sim_ref` false).

**`anchor_wer` is not the headline number.** It is a per-utterance rate, and
averaging it gives the *macro* aggregation that v2 keeps only for continuity with
v1. The headline is corpus-level, and the counts are shipped so it is derivable
from the dataset alone:

```python
import datasets
d = datasets.load_dataset("nineninesix/multilingual-speech-benchmark", "en-US", split="main").to_pandas()

wer_corpus = (d.anchor_subs + d.anchor_dels + d.anchor_ins).sum() \
             / d.anchor_n_ref_words.sum()      # 0.0774 — the headline
wer_macro  = d.anchor_wer.mean()               # 0.0807 — v1 legacy
```

The same holds for any slice: filter the rows first, then sum. Averaging
`anchor_wer` over a slice silently switches you back to the macro form.

One further caution: the SIM anchor is **constant within a speaker** by
construction, so those columns are not independent observations. Cluster by
`speaker_id` before putting an interval on anything.

**Identity:** `utt` (stable join key), `lang`, `subset`, `source`, `cv_version`.

**Prompt:** `prompt_text`, `prompt_dur` (after trimming), `prompt_sr_orig`,
`prompt_dur_bin`, `prompt_cv_path`, `prompt_split_origin`.

**Speaker:** `speaker_id`, `speaker_gender` (+ `_source`), `speaker_age`,
`speaker_accent_label` (raw label, secondary accents included),
`speaker_n_in_subset`.

**Target text:** `text`, `text_norm` (produced by `msbench.normalize`, not a
placeholder), `n_words`, `n_chars`, `len_bin`, `phones` (IPA),
`n_phones`, `punct_type`, `text_cv_path`, `text_split_origin`.

**Anchor availability:** `has_gt`, `gt_dur`, `gt_speaker_id`, `gt_same_speaker`,
`gt_gender`, `has_sim_ref`, `sim_ref_dur`, `sim_ref_dur_bin`, `sim_ref_text`.

**Prompt QC** (`qc_*`, on the untrimmed clip): `qc_vad_speech_ratio`,
`qc_lead_sil`, `qc_trail_sil`, `qc_clip_rate`, `qc_hf_energy_db` (dB above
5 kHz), `qc_bandwidth_hz` (deprecated alias of the previous), `qc_dnsmos_ovrl`,
`qc_dnsmos_sig`, `qc_dnsmos_bak`, `qc_spk_centroid_dist`.

**Ground-truth and second-clip QC** (new in v2, reporting only): `gt_qc_*` and
`simref_qc_*`, each with `dnsmos_ovrl`, `dnsmos_sig`, `dnsmos_bak`, `clip_rate`,
`hf_energy_db`, `vad_speech_ratio`, `lead_sil`, `trail_sil`, `fails_prompt_qc`.

**Reserved for a future hard set**, currently constant: `category` (`general`),
`subcategory`, `tags` (empty `list<string>`), `difficulty` (0), `has_digit`,
`has_abbrev`, `has_foreign` (all false), `notes`.

## Reports

`reports/` is part of the release, not an afterthought.

| path | contents |
|---|---|
| `reports/results.md` | all anchors with intervals, breakdowns with per-cell `n` |
| `reports/attribution.md` | the v1 → v2 ladder, one cause per delta |
| `reports/summary.md`, `reports/coverage_*.md` | per-subset composition and coverage |
| `reports/quality_{prompt,gt,sim_ref}.md` | DNSMOS and QC per audio stream |
| `reports/csv/*.csv` | every table above, machine-readable |
| `reports/per_utterance/*.parquet` | **per-`utt` metrics for every recogniser and encoder**, with S/D/I counts, reference lengths, both normalised strings and `speaker_id` |
| `reports/per_utterance/*.json` | the exact configuration of every run |

The per-utterance artifacts are what let you re-aggregate any number in this
card, audit outliers, run your own paired tests, or slice by any column — without
a GPU and without re-running a recogniser. v1 shipped aggregate markdown only.

## Limitations

1. **No naturalness axis.** Neither subjective nor predicted. A system can score
   well here and sound robotic.
2. **No hard set.** Common Voice contains 0 % texts with digits, so text
   normalisation is untested.
3. **No long-form.** Texts are 3-16 words; long-context prosody and stability are
   untested.
4. **`ky` rests on 89 ground-truth speakers** for 694 rows, with a correspondingly
   wide interval.
5. **The WER anchor mixes three error sources** — recogniser error, reader error,
   and unverified crowd transcripts — with no way to separate them.
6. **Whisper has plausibly seen Common Voice**, which flatters the anchor in the
   opposite direction to the previous point. Neither bias is quantified.
7. **The QC filter and v1's SIM encoder share an embedding space.** Use `ecapa`
   to see past it.
8. **v1's SIM encoder has almost no usable range**, and on `ky` its impostor p95
   exceeds the human anchor.
9. **Observations are not independent**; always cluster by speaker.
10. **Ground-truth and second-clip audio were never rejection-filtered**;
    1.2-4.3 % would fail the prompt criteria.
11. **`utt` stability was prioritised over data cleanliness.** Nothing is
    filtered on the new QC columns.
12. **The macro→corpus change makes v2 numbers incomparable with v1's** unless you
    use the `wer_macro` column, which is retained for exactly that purpose.
13. **The repetition-penalty removal is justified theoretically**, not
    empirically: its effect on synthesis has not been measured here.
14. **Cross-language comparison of anchors is not supported.**
15. **Packaged audio was resampled to 16 kHz with linear interpolation at build
    time**, before v2 fixed the evaluation path. That is frozen into the data and
    is one more reason not to compare absolute SIM across datasets.
16. **Prompt durations are 2.5-5 s**, shorter than the 3-20 s Seed-TTS uses, and
    SIM rises with prompt duration — so even same-encoder comparison with that
    literature is only partial.
17. **Speaker metadata is self-declared** and missing for a large share of
    speakers (44 % of `ky` gender).
18. **pt-BR splits are broken upstream**: 97 % of test sentences also appear in
    train, and 9,464 clips are duplicated between train and dev.
19. **`n_syllables` was dropped rather than approximated** — every heuristic
    broke on hiatus or diphthongs. Use `n_phones / gt_dur` for speech rate.
20. **Scribe is a paid, hosted, non-deterministic recogniser.** It is the most
    accurate reader of these recordings on five subsets, but it cannot be the
    primary anchor of an open benchmark: reproducing it costs money, needs
    network access, and lands within ~0.002 rather than exactly. Whisper and
    GigaAM stay primary for that reason, not because they are better.
21. **DNSMOS is a reporting metric here, never a filter.** Reference quality was
    measured against SIM and explains under 1.6 % of its variance, so it is not a
    confounder — but that was measured on the human anchor, where noise affects
    both embeddings and partly cancels.

## Reproduction

```bash
git clone https://github.com/nineninesix-ai/make-speech-benchmark && cd make-speech-benchmark
uv venv && source .venv/bin/activate && uv pip install -e ".[all]"

msbench-fetch                  # download this dataset
bash scripts/runbook.sh all    # every number in this card
msbench-validate               # v2 differs from v1 only where claimed
pytest                         # the aggregation gate
```

Selection stages S1-S3 are **not** re-run and are not needed for any of the
above; they require the 66 GB Common Voice corpus and would change `utt`.

Two documents in the repository carry more detail than fits here:
[`docs/PROTOCOL.md`](https://github.com/nineninesix-ai/make-speech-benchmark/blob/main/docs/PROTOCOL.md) is the full measurement
contract, and [`docs/DEFECTS.md`](https://github.com/nineninesix-ai/make-speech-benchmark/blob/main/docs/DEFECTS.md) is the
catalogue of all 27 v1 defects with what each one cost.

## Personal and sensitive information

The audio is human speech from Common Voice, contributed under CC0 by volunteers
who consented to public release. Speaker identifiers are truncated Common Voice
`client_id` hashes and are not linkable to a person by this dataset alone.
Demographic fields are self-declared and frequently absent. The recordings are
nonetheless **biometric voice data**: they can be used to build speaker models,
and this dataset exists to measure exactly that capability. Do not use them to
impersonate the contributors.

Sentences come from Common Voice's own text corpora and may contain the usual
errors of crowd-sourced transcription; several were identified during this work
where the reference text disagrees with what was actually said.

## Licence, citation, contact

**CC0-1.0**, matching Common Voice 17.0.

```bibtex
@misc{multilingual_speech_benchmark_2026,
  title  = {Multilingual Speech Benchmark for Zero-Shot TTS},
  author = {nineninesix},
  year   = {2026},
  note   = {Version 2.0},
  url    = {https://huggingface.co/datasets/nineninesix/multilingual-speech-benchmark}
}
```

Pipeline: [https://github.com/nineninesix-ai/make-speech-benchmark](https://github.com/nineninesix-ai/make-speech-benchmark). Issues and pull requests welcome — adding a language,
a recogniser backend or a speaker encoder each touch one file.

## References

- Common Voice 17.0 — <https://commonvoice.mozilla.org>
- Seed-TTS ([arXiv:2406.02430](https://arxiv.org/abs/2406.02430)) and
  seed-tts-eval, the source of the corpus-level WER convention and the
  `wavlm_large_finetune.pth` SIM checkpoint
- Whisper ([arXiv:2212.04356](https://arxiv.org/abs/2212.04356))
- MMS ([arXiv:2305.13516](https://arxiv.org/abs/2305.13516))
- WavLM ([arXiv:2110.13900](https://arxiv.org/abs/2110.13900))
- ECAPA-TDNN ([arXiv:2005.07143](https://arxiv.org/abs/2005.07143))
- DNSMOS P.835 ([arXiv:2110.01763](https://arxiv.org/abs/2110.01763))
- Silero VAD — <https://github.com/snakers4/silero-vad>

import re

with open("src/main.py", "r", encoding="utf-8") as f:
    main_content = f.read()

# Add AnalysisContext import
if "AnalysisContext" not in main_content:
    main_content = main_content.replace(
        "from src.vlm.video_chunker import VideoChunk",
        "from src.vlm.video_chunker import VideoChunk, AnalysisContext"
    )

old_run = '''    def run(
        self,
        video_source: str,
        user_prompt: str,
        on_stage: Optional[OnStageCallback] = None,
        sample_fps_override: Optional[int] = None,
        min_change_threshold_override: Optional[float] = None,
        trace: Optional[TraceCallback] = None,
        analysis_mode: str = "vlm_direct",
    ) -> SafirReport:'''
new_run = '''    def run(
        self,
        video_source: str,
        user_prompt: str,
        on_stage: Optional[OnStageCallback] = None,
        sample_fps_override: Optional[int] = None,
        min_change_threshold_override: Optional[float] = None,
        trace: Optional[TraceCallback] = None,
        analysis_mode: str = "vlm_direct",
        context: Optional[AnalysisContext] = None,
    ) -> SafirReport:'''
main_content = main_content.replace(old_run, new_run)

# Update stage_vlm definition
old_stage_vlm = '''    def stage_vlm(
        self,
        video_source: str,
        evidence_frames: List[EvidenceFrame],
        user_prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
        analysis_mode: str = "vlm_direct",
    ) -> VLMResponse:'''
new_stage_vlm = '''    def stage_vlm(
        self,
        video_source: str,
        evidence_frames: List[EvidenceFrame],
        user_prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
        analysis_mode: str = "vlm_direct",
        context: Optional[AnalysisContext] = None,
    ) -> VLMResponse:'''
main_content = main_content.replace(old_stage_vlm, new_stage_vlm)

old_stage_vlm_return = '''        if getattr(vlm, "requires_frame_sampling", True):
            return self._stage_vlm_frames(vlm, evidence_frames, user_prompt)
        return self._stage_vlm_video(vlm, video_source, evidence_frames, user_prompt, on_progress)'''
new_stage_vlm_return = '''        if getattr(vlm, "requires_frame_sampling", True):
            return self._stage_vlm_frames(vlm, evidence_frames, user_prompt)
        return self._stage_vlm_video(vlm, video_source, evidence_frames, user_prompt, on_progress, context)'''
main_content = main_content.replace(old_stage_vlm_return, new_stage_vlm_return)

old_stage_vlm_video = '''    def _stage_vlm_video(
        self,
        vlm: BaseVLM,
        video_source: str,
        evidence_frames: List[EvidenceFrame],
        user_prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
    ) -> VLMResponse:'''
new_stage_vlm_video = '''    def _stage_vlm_video(
        self,
        vlm: BaseVLM,
        video_source: str,
        evidence_frames: List[EvidenceFrame],
        user_prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
        context: Optional[AnalysisContext] = None,
    ) -> VLMResponse:'''
main_content = main_content.replace(old_stage_vlm_video, new_stage_vlm_video)

old_analyze_call = '''            try:
                response = analyze(video_source, evidence_frames, prompt=user_prompt, on_progress=on_progress)
            except TypeError:
                # Geriye-donuk uyumluluk: `on_progress` desteklemeyen bir'''
new_analyze_call = '''            try:
                response = analyze(video_source, evidence_frames, prompt=user_prompt, on_progress=on_progress, context=context)
            except TypeError:
                # Geriye-donuk uyumluluk: `on_progress` desteklemeyen bir'''
main_content = main_content.replace(old_analyze_call, new_analyze_call)

old_run_stage_vlm_call = '''        vlm_response = self.stage_vlm(
            video_source, evidence_frames, user_prompt, on_vlm_progress, analysis_mode
        )'''
new_run_stage_vlm_call = '''        vlm_response = self.stage_vlm(
            video_source, evidence_frames, user_prompt, on_vlm_progress, analysis_mode, context
        )'''
main_content = main_content.replace(old_run_stage_vlm_call, new_run_stage_vlm_call)

# Update _background_analyze to pass context
old_bg_analyze = '''def _background_analyze(
    job_id: str, request: AnalyzeRequest, pipeline: SafirPipeline, video_source: str
) -> None:'''
new_bg_analyze = '''def _background_analyze(
    job_id: str, request: AnalyzeRequest, pipeline: SafirPipeline, video_source: str
) -> None:
    context = AnalysisContext(analysis_id=job_id, video_id=str(uuid.uuid4()))'''
main_content = main_content.replace(old_bg_analyze, new_bg_analyze)

old_bg_analyze_run = '''            report = pipeline.run(
                video_source,
                request.prompt,
                on_stage=on_stage_callback,
                sample_fps_override=request.sample_fps,
                min_change_threshold_override=request.min_change_threshold,
                trace=trace_callback,
                analysis_mode=request.analysis_mode,
            )'''
new_bg_analyze_run = '''            report = pipeline.run(
                video_source,
                request.prompt,
                on_stage=on_stage_callback,
                sample_fps_override=request.sample_fps,
                min_change_threshold_override=request.min_change_threshold,
                trace=trace_callback,
                analysis_mode=request.analysis_mode,
                context=context,
            )'''
main_content = main_content.replace(old_bg_analyze_run, new_bg_analyze_run)

# Update sync analyze to pass context
old_analyze_sync = '''def analyze_video(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> Union[SafirReport, JobStatusResponse]:'''
new_analyze_sync = '''def analyze_video(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> Union[SafirReport, JobStatusResponse]:
    context = AnalysisContext(analysis_id=str(uuid.uuid4()), video_id=str(uuid.uuid4()))'''
main_content = main_content.replace(old_analyze_sync, new_analyze_sync)

old_sync_run = '''        return pipeline.run(
            video_source,
            request.prompt,
            sample_fps_override=request.sample_fps,
            min_change_threshold_override=request.min_change_threshold,
            analysis_mode=request.analysis_mode,
        )'''
new_sync_run = '''        return pipeline.run(
            video_source,
            request.prompt,
            sample_fps_override=request.sample_fps,
            min_change_threshold_override=request.min_change_threshold,
            analysis_mode=request.analysis_mode,
            context=context,
        )'''
main_content = main_content.replace(old_sync_run, new_sync_run)


with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(main_content)

print("main.py refactored.")

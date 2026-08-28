import sys
content = '''<script setup lang="ts">
import { useRoute } from "vue-router"
import { useAnalysisDetail } from "~/composables/useAnalysisDetail"
import { ref } from "vue"
import ReportVideoViewer from "~/components/operator/report/ReportVideoViewer.vue"
import ReportRiskCard from "~/components/operator/report/ReportRiskCard.vue"
import BackgroundScene from "~/components/BackgroundScene.vue"
import ReportDownloadButton from "~/components/operator/report/ReportDownloadButton.vue"
import AskSafirDrawer from "~/components/operator/report/AskSafirDrawer.vue"

definePageMeta({ layout: "operator" })

const route = useRoute()
const jobId = route.params.id as string

const { state, report, jobStatus, step, totalSteps, errorMsg, reload } = useAnalysisDetail(jobId)

export interface AskSafirDrawerHandle { open: () => void; close: () => void }
const askSafirRef = ref<AskSafirDrawerHandle | null>(null)

function openAskSafir() {
  askSafirRef.value?.open()
}

function getBasename(path: string | null) {
  if (!path) return "Video bilgisi bulunmuyor"
  return path.split(/[/\\]/).pop() || "Video"
}

function formatDate(dateStr: string) {
  if (!dateStr) return ""
  try {
    return new Intl.DateTimeFormat("tr-TR", { 
      dateStyle: "medium", 
      timeStyle: "short" 
    }).format(new Date(dateStr))
  } catch {
    return dateStr
  }
}

function formatDateUpper(dateStr: string) {
  return formatDate(dateStr).toUpperCase()
}

function formatRiskLevel(level: string | null) {
  if (!level) return "BÝLGÝ YOK"
  const l = level.toLowerCase()
  if (l === "dusuk" || l === "low") return "DÜÞÜK"
  if (l === "orta" || l === "medium") return "ORTA"
  if (l === "yuksek" || l === "high") return "YÜKSEK"
  if (l === "kritik" || l === "critical") return "KRÝTÝK"
  return "BÝLGÝ YOK"
}
</script>

'''
with open('new_analizler.vue', 'w', encoding='utf-8') as f:
  f.write(content)

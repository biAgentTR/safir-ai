import { ref, onBeforeUnmount } from 'vue'
import { isTauri } from '@tauri-apps/api/core'
import { save } from '@tauri-apps/plugin-dialog'
import { writeFile } from '@tauri-apps/plugin-fs'
import { useSafirApi } from './useSafirApi'

export type ReportDownloadStatus =
  | 'idle'
  | 'fetching'
  | 'choosing-location'
  | 'saving'
  | 'success'
  | 'error'

export function useReportDownload() {
  const api = useSafirApi()
  const statusMap = ref<Map<string, ReportDownloadStatus>>(new Map())
  const errorMessage = ref<string>('')
  
  const controllers = new Map<string, AbortController>()

  function setStatus(jobId: string, status: ReportDownloadStatus) {
    statusMap.value.set(jobId, status)
  }

  function getStatus(jobId: string): ReportDownloadStatus {
    return statusMap.value.get(jobId) || 'idle'
  }

  function cancelCurrentRequest(jobId: string) {
    if (controllers.has(jobId)) {
      controllers.get(jobId)?.abort()
      controllers.delete(jobId)
    }
  }

  onBeforeUnmount(() => {
    for (const jobId of controllers.keys()) {
      cancelCurrentRequest(jobId)
    }
  })

  async function downloadReport(jobId: string) {
    const currentStatus = getStatus(jobId)
    if (currentStatus === 'fetching' || currentStatus === 'choosing-location' || currentStatus === 'saving') {
      return
    }
    
    cancelCurrentRequest(jobId)
    const abortController = new AbortController()
    controllers.set(jobId, abortController)
    
    setStatus(jobId, 'fetching')
    errorMessage.value = ''
    
    try {
      const response = await api.downloadHistoryReport(jobId, abortController.signal)
      
      if (isTauri()) {
        setStatus(jobId, 'choosing-location')
        const filePath = await save({
          defaultPath: response.filename,
          filters: [{ name: 'PDF Belgesi', extensions: ['pdf'] }]
        })
        
        if (!filePath) {
          // User cancelled dialog
          setStatus(jobId, 'idle')
          return
        }
        
        setStatus(jobId, 'saving')
        await writeFile(filePath, response.bytes)
        setStatus(jobId, 'success')
      } else {
        // Browser fallback
        setStatus(jobId, 'saving')
        const buffer = new ArrayBuffer(response.bytes.byteLength)
        const view = new Uint8Array(buffer)
        view.set(new Uint8Array(response.bytes.buffer, response.bytes.byteOffset, response.bytes.byteLength))
        const blob = new Blob([buffer], { type: 'application/pdf' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = response.filename
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        setTimeout(() => URL.revokeObjectURL(url), 1000)
        setStatus(jobId, 'success')
      }
      
      // Auto reset success state after 3s
      setTimeout(() => {
        if (getStatus(jobId) === 'success') {
          setStatus(jobId, 'idle')
        }
      }, 3000)
      
    } catch (err: any) {
      if (err.name === 'AbortError' || err.message?.includes('aborted')) {
        setStatus(jobId, 'idle')
        return
      }
      
      setStatus(jobId, 'error')
      
      if (err.response?.status === 404) {
        errorMessage.value = 'Rapor bulunamadı. Bu analiz kaydı kaldırılmış veya henüz rapor oluşturulmamış olabilir.'
      } else if (err.response?.status === 503) {
        errorMessage.value = 'PDF servisi kullanılamıyor. Rapor oluşturma bileşeni şu anda kullanılamıyor.'
      } else if (err.response?.status >= 500 || err.message?.includes('fetch')) {
        errorMessage.value = 'Rapor servisine ulaşılamıyor.'
      } else if (getStatus(jobId) === 'saving') {
        errorMessage.value = 'PDF seçtiğiniz konuma kaydedilemedi.'
      } else {
        errorMessage.value = err.message || 'Geçerli bir PDF raporu alınamadı.'
      }
      
      setTimeout(() => {
        if (getStatus(jobId) === 'error') {
          setStatus(jobId, 'idle')
        }
      }, 5000)
    } finally {
      controllers.delete(jobId)
    }
  }

  return {
    getStatus,
    errorMessage,
    downloadReport
  }
}
export type DrawerSlideId = 'gecmis' | 'asistan' | 'raporlar'

const activeSlide = ref<DrawerSlideId>('asistan')
const isTransitioning = ref(false)

export function useDrawerDeck() {
  function setSlide(id: DrawerSlideId) {
    activeSlide.value = id
  }

  function setTransitioning(val: boolean) {
    isTransitioning.value = val
  }

  return {
    activeSlide,
    isTransitioning,
    setSlide,
    setTransitioning,
  }
}

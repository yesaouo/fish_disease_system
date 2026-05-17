import { useContext, useEffect } from "react";
import { UNSAFE_NavigationContext as NavigationContext } from "react-router";

export const useNavBlocker = (when: boolean, bypass?: (nextLocation: any) => boolean) => {
  const { navigator } = useContext(NavigationContext) as any;
  useEffect(() => {
    if (!when || !navigator?.block) return;
    const unblock = navigator.block((tx: any) => {
      if (bypass?.(tx.location)) {
        unblock();
        tx.retry();
        return;
      }
      const ok = window.confirm("目前有未儲存的變更，離開將放棄這些變更。確定要離開嗎？");
      if (ok) {
        unblock();
        tx.retry();
      }
    });
    return unblock;
  }, [when, navigator, bypass]);
};

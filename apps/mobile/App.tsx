import { useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { StyleSheet } from 'react-native';
import { RootNavigator } from './src/navigation';
import { usePreferencesStore } from './src/store/usePreferencesStore';

export default function App() {
  const initRegion = usePreferencesStore((s) => s.initRegionFromLocale);
  useEffect(() => { initRegion(); }, [initRegion]);

  return (
    <GestureHandlerRootView style={styles.container}>
      <RootNavigator />
      <StatusBar style="auto" />
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

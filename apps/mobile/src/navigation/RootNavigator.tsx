import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import MainTabNavigator from './MainTabNavigator';
import { DetectionScreen, VlmDownloadScreen, GeminiNanoTestScreen } from '../screens';

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function RootNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
        }}
      >
        <Stack.Screen name="Main" component={MainTabNavigator} />
        <Stack.Screen
          name="Detection"
          component={DetectionScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
          }}
        />
        <Stack.Screen
          name="VlmDownload"
          component={VlmDownloadScreen}
          options={{
            headerShown: true,
            headerTitle: 'VLM Model',
            animation: 'slide_from_right',
          }}
        />
        <Stack.Screen
          name="GeminiNanoTest"
          component={GeminiNanoTestScreen}
          options={{
            headerShown: true,
            headerTitle: 'Gemini Nano Test',
            animation: 'slide_from_right',
          }}
        />
        {/* TODO: Add EntryDetail screen */}
      </Stack.Navigator>
    </NavigationContainer>
  );
}

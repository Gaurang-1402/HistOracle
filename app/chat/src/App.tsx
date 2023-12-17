import './App.css';

import {
  AdditionalPhonemeInfo,
  Character,
  EmotionEvent,
  HistoryItem,
  InworldConnectionService,
  InworldPacket,
} from '@inworld/web-sdk';
import { Button } from '@mui/material';
import { useCallback, useEffect, useRef, useState } from 'react';
import { FormProvider, useForm } from 'react-hook-form';
import { useHistory, useParams } from 'react-router-dom';

import { Chat } from './app/chat/Chat';
import { PlayerNameInput } from './app/chat/Chat.styled';
import { Avatar } from './app/components/3dAvatar/Avatar';
import { Layout } from './app/components/Layout';
import {
  ChatWrapper as StyledChatWrapper,
  MainWrapper as StyledMainWrapper,
} from './app/components/Simulator';
import { InworldService } from './app/connection';
import { save as saveConfiguration } from './app/helpers/configuration';
import { CHAT_VIEW, Configuration, EmotionsMap } from './app/types';
import { config } from './config';
import * as defaults from './defaults';

interface CurrentContext {
  allCharacters: Character[];
  isChatting: boolean;
  activeConnection?: InworldConnectionService;
}

interface RouteParams {
  characterName: string;
}

function App() {
  const formMethods = useForm<Configuration>({ mode: 'onChange' });
  const [currentPlayerName, setCurrentPlayerName] = useState<string | null>(null);

  const [activeConnection, setActiveConnection] = useState<InworldConnectionService>();
  const [activeCharacter, setActiveCharacter] = useState<Character>();
  const [allCharacters, setAllCharacters] = useState<Character[]>([]);
  const [chatHistory, setChatHistory] = useState<HistoryItem[]>([]);
  const [isChatting, setIsChatting] = useState(false);
  const [currentChatView, setCurrentChatView] = useState(CHAT_VIEW.TEXT);
  const [currentPhonemes, setCurrentPhonemes] = useState<AdditionalPhonemeInfo[]>([]);
  const [currentEmotionEvent, setCurrentEmotionEvent] = useState<EmotionEvent>();
  const [currentEmotions, setCurrentEmotions] = useState<EmotionsMap>({});

  const contextRef = useRef<CurrentContext>();
  contextRef.current = {
    allCharacters,
    isChatting,
    activeConnection,
  };

  const handleHistoryChange = useCallback((history: HistoryItem[]) => {
    setChatHistory(history);
  }, []);

  const establishConnection = useCallback(async () => {
    if (currentPlayerName === null) {
      console.log('Set player name first!');
      return;
    }
    console.log('Retrying!');
    const formValues = formMethods.getValues();
    console.log(formValues);

    setIsChatting(true);
    setCurrentChatView(formValues.chatView!);

    const service = new InworldService({
      onHistoryChange: handleHistoryChange,
      capabilities: {
        ...(formValues.chatView === CHAT_VIEW.AVATAR && { phonemes: true }),
        emotions: true,
        narratedActions: true,
      },
      sceneName: formValues.scene?.name!,
      playerName: currentPlayerName,
      onPhoneme: (phonemes: AdditionalPhonemeInfo[]) => {
        setCurrentPhonemes(phonemes);
      },
      onReady: async () => {
        console.log('Ready!');
      },
      onDisconnect: () => {
        console.log('Disconnected!');
      },
      onMessage: (inworldPacket: InworldPacket) => {
        if (
          inworldPacket.isEmotion() &&
          inworldPacket.packetId?.interactionId
        ) {
          setCurrentEmotionEvent(inworldPacket.emotions);
          setCurrentEmotions((currentState) => ({
            ...currentState,
            [inworldPacket.packetId.interactionId]: inworldPacket.emotions,
          }));
        }
      },
    });

    const charactersList = await service.connection.getCharacters();
    const currentCharacter = charactersList.find(
      (c: Character) => c.resourceName === formValues.character?.name,
    );

    if (currentCharacter) {
      service.connection.setCurrentCharacter(currentCharacter);
    }

    setActiveConnection(service.connection);
    setActiveCharacter(currentCharacter);
    setAllCharacters(charactersList);
  }, [formMethods, handleHistoryChange, currentPlayerName]);

  const browserHistory = useHistory();

  const concludeChatting = useCallback(async () => {

    // Disable flags
    setIsChatting(false);

    // Stop audio playback and recording
    activeConnection?.player?.stop();
    activeConnection?.player?.clear();
    activeConnection?.recorder?.stop();

    // Clear collections
    setChatHistory([]);

    // Close connection and clear connection data
    activeConnection?.close();
    setActiveConnection(undefined);
    setActiveCharacter(undefined);
    setAllCharacters([]);

    // Navigate back to the home page
    browserHistory.push('/');
  }, [activeConnection, browserHistory]);

  const resetFormValues = useCallback(() => {
    formMethods.reset({
      ...defaults.configuration,
    });
    saveConfiguration(formMethods.getValues());
  }, [formMethods]);

  const submitPlayerName = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const playerNameInput = e.currentTarget.elements.namedItem(
      'playerName',
    ) as HTMLInputElement;
    setCurrentPlayerName(playerNameInput.value);
  };

  useEffect(() => {
    if (currentPlayerName) {
      establishConnection();
    }
  }, [currentPlayerName, establishConnection]);

  const { characterName } = useParams<RouteParams>();

  useEffect(() => {
    if (browserHistory && browserHistory.location && characterName) {
      formMethods.setValue(
        'scene.name',
        `workspaces/default-_wue38lymyd9iehiz6gbng/scenes/${characterName}_statue`,
      );
      formMethods.setValue(
        'character.name',
        `workspaces/default-_wue38lymyd9iehiz6gbng/characters/${characterName}`,
      );
    }
  }, [browserHistory, formMethods, characterName]);

  const applicationContent = (
    <>
      {currentPlayerName ? (
        activeCharacter ? (
          <StyledMainWrapper>
            <StyledChatWrapper>
              <Avatar
                emotionEvent={currentEmotionEvent}
                phonemes={currentPhonemes}
                visible={currentChatView === CHAT_VIEW.AVATAR}
                url={
                  config.RPM_AVATAR ||
                  activeCharacter.assets.rpmModelUri ||
                  defaults.DEFAULT_RPM_AVATAR
                }
              />
              <Chat
                chatView={currentChatView}
                chatHistory={chatHistory}
                connection={activeConnection!}
                emotions={currentEmotions}
              />
            </StyledChatWrapper>
          </StyledMainWrapper>
        ) : (
          'Loading...'
        )
      ) : (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100vh',
          }}
        >
          <form onSubmit={submitPlayerName}>
            <label style={{ color: 'white' }}>
              Enter your name:
              <PlayerNameInput type="text" name="playerName" required />
            </label>
            <Button type="submit" variant="contained">
              Submit
            </Button>
          </form>
        </div>
      )}
    </>
  );

  return (
    <FormProvider {...formMethods}>
      <Layout>{applicationContent}</Layout>
    </FormProvider>
  );
}

export default App;
